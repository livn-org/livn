/* Stimulus delivery, in the four modes the Python layer hands over.
 *
 * The timing mirrors the NEURON backend: an extracellular field is turned
 * into equivalent axial currents by a per-step callback that runs after the
 * voltage update, so the field sampled at step s drives step s + 1; a
 * photon flux (or, for ChR2H134R, irradiance) set by the same callback enters
 * the opsin kinetics of step s;
 * a current-clamp amplitude is what Vector.play interpolates at the step
 * midpoint; a current density is held over its own sample.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"

void rcsd_stimulus_free(Stimulus* s) {
    free(s->cell);
    free(s->section);
    free(s->node);
    free(s->values);
    free(s->j_row_a);
    free(s->j_row_b);
    free(s->j_node_a);
    free(s->j_node_b);
    free(s->j_inv_r);
    memset(s, 0, sizeof *s);
    s->needed = -1;
}

/* half of a segment's axial resistance, in MOhm (cells.half_axial_resistance) */
static double half_axial(const Section* sec) {
    double half_length_cm = (sec->L / (double) sec->nseg) * 1e-4 / 2.0;
    double radius_cm = sec->diam * 1e-4 / 2.0;
    if (radius_cm <= 0.0) {
        return 0.0;
    }
    return sec->Ra * half_length_cm / (M_PI * radius_cm * radius_cm) * 1e-6;
}

/* the first row naming each section, by global section index, -1 for none:
 * one pass over the rows instead of a scan of them per junction, which made
 * building the junctions quadratic in the network (120 s at 100k cells) */
static int* first_rows(const RCSDSim* sim, const Stimulus* s) {
    int* first = (int*) malloc((sim->sections.n + 1) * sizeof(int));
    size_t i;
    int r;
    if (first == NULL) {
        return NULL;
    }
    for (i = 0; i < sim->sections.n; ++i) {
        first[i] = -1;
    }
    for (r = s->n_rows - 1; r >= 0; --r) {
        const Cell* cell;
        if (s->cell[r] < 0 || (size_t) s->cell[r] >= sim->cells.n) {
            continue;
        }
        cell = &sim->cells.data[s->cell[r]];
        if (s->section[r] < 0 || s->section[r] >= cell->nsec) {
            continue;
        }
        first[cell->sec0 + s->section[r]] = r;
    }
    return first;
}

/* the junctions of every cell the field reaches on both sides */
static int build_junctions(RCSDSim* sim, Stimulus* s) {
    size_t sec_index;
    int cap = 0, n = 0, dropped = 0;
    int* first;
    free(s->j_row_a);
    free(s->j_row_b);
    free(s->j_node_a);
    free(s->j_node_b);
    free(s->j_inv_r);
    s->j_row_a = s->j_row_b = s->j_node_a = s->j_node_b = NULL;
    s->j_inv_r = NULL;
    first = first_rows(sim, s);
    if (first == NULL) {
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    for (sec_index = 0; sec_index < sim->sections.n; ++sec_index) {
        const Section* child = &sim->sections.data[sec_index];
        const Section* parent;
        int row_a, row_b, node_a, node_b, j;
        double resistance;
        if (child->parent_section < 0) {
            continue;
        }
        parent = &sim->sections.data[child->parent_section];
        resistance = half_axial(child) + half_axial(parent);
        if (resistance <= 0.0) {
            continue;
        }
        row_a = first[sec_index];
        row_b = first[child->parent_section];
        if (row_a < 0 || row_b < 0) {
            dropped += 1;
            continue;
        }
        node_a = child->node0;
        j = (int) (child->parent_x * parent->nseg);
        if (j < 0) {
            j = 0;
        }
        if (j >= parent->nseg) {
            j = parent->nseg - 1;
        }
        node_b = parent->node0 + j;
        if (n == cap) {
            int* ra;
            int* rb;
            int* na;
            int* nb;
            double* ir;
            cap = cap ? cap * 2 : 64;
            ra = (int*) realloc(s->j_row_a, (size_t) cap * sizeof(int));
            rb = (int*) realloc(s->j_row_b, (size_t) cap * sizeof(int));
            na = (int*) realloc(s->j_node_a, (size_t) cap * sizeof(int));
            nb = (int*) realloc(s->j_node_b, (size_t) cap * sizeof(int));
            ir = (double*) realloc(s->j_inv_r, (size_t) cap * sizeof(double));
            if (!ra || !rb || !na || !nb || !ir) {
                free(first);
                rcsd_set_error("out of memory");
                return RCSD_ERROR;
            }
            s->j_row_a = ra;
            s->j_row_b = rb;
            s->j_node_a = na;
            s->j_node_b = nb;
            s->j_inv_r = ir;
        }
        s->j_row_a[n] = row_a;
        s->j_row_b[n] = row_b;
        s->j_node_a[n] = node_a;
        s->j_node_b[n] = node_b;
        s->j_inv_r[n] = 1.0 / resistance;
        n += 1;
    }
    free(first);
    s->n_junctions = n;
    s->dropped_junctions = dropped;
    return RCSD_OK;
}

int rcsd_set_stimulus_rows(RCSDSim* sim, int mode, int n_rows, const int* cell,
                           const int* section, double stim_dt) {
    Stimulus* s;
    int r;
    if (mode < 0 || mode >= RCSD_STIM_N) {
        rcsd_set_error("no stimulus mode %d", mode);
        return RCSD_ERROR;
    }
    if (!(stim_dt > 0.0)) {
        rcsd_set_error("stimulus dt must be positive");
        return RCSD_ERROR;
    }
    s = &sim->stim[mode];
    rcsd_stimulus_free(s);
    s->cell = (int*) malloc((size_t) (n_rows > 0 ? n_rows : 1) * sizeof(int));
    s->section = (int*) malloc((size_t) (n_rows > 0 ? n_rows : 1) * sizeof(int));
    s->node = (int*) malloc((size_t) (n_rows > 0 ? n_rows : 1) * sizeof(int));
    if (!s->cell || !s->section || !s->node) {
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    for (r = 0; r < n_rows; ++r) {
        int sec = rcsd_cell_section(sim, cell[r], section[r]);
        if (sec < 0) {
            rcsd_set_error("stimulus row %d: cell %d has no section %d", r, cell[r], section[r]);
            return RCSD_ERROR;
        }
        s->cell[r] = cell[r];
        s->section[r] = section[r];
        s->node[r] = rcsd_section_node(sim, sec, 0.5);
    }
    s->n_rows = n_rows;
    s->stim_dt = stim_dt;
    s->active = 1;
    s->extent = 0;
    s->first = 0;
    s->n_samples = 0;
    s->needed = -1;
    if (mode == RCSD_STIM_EXTRACELLULAR) {
        return build_junctions(sim, s);
    }
    return RCSD_OK;
}

int rcsd_set_stimulus_window(RCSDSim* sim, int mode, long first, int n_samples,
                             const double* values, long extent) {
    Stimulus* s;
    double* copy;
    if (mode < 0 || mode >= RCSD_STIM_N) {
        rcsd_set_error("no stimulus mode %d", mode);
        return RCSD_ERROR;
    }
    s = &sim->stim[mode];
    if (!s->active) {
        rcsd_set_error("declare the stimulus rows before handing over a window");
        return RCSD_ERROR;
    }
    if (n_samples < 0) {
        n_samples = 0;
    }
    copy = (double*) malloc((size_t) (n_samples * s->n_rows > 0 ? n_samples * s->n_rows : 1) *
                            sizeof(double));
    if (copy == NULL) {
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    if (n_samples > 0 && s->n_rows > 0) {
        memcpy(copy, values, (size_t) n_samples * (size_t) s->n_rows * sizeof(double));
    }
    free(s->values);
    s->values = copy;
    s->first = first;
    s->n_samples = n_samples;
    s->extent = extent;
    s->needed = -1;
    return RCSD_OK;
}

int rcsd_clear_stimulus(RCSDSim* sim, int mode) {
    int i;
    if (mode < 0 || mode >= RCSD_STIM_N) {
        rcsd_set_error("no stimulus mode %d", mode);
        return RCSD_ERROR;
    }
    rcsd_stimulus_free(&sim->stim[mode]);
    if (mode == RCSD_STIM_EXTRACELLULAR) {
        RCSD_ACC_LOOP(sim)
        for (i = 0; i < sim->n_nodes; ++i) {
            sim->ext_amp[i] = 0.0;
        }
    }
    if (mode == RCSD_STIM_PHOTON_FLUX) {
        size_t o;
        for (o = 0; o < sim->opsins.n; ++o) {
            sim->opsins.data[o].phi = 0.0;
        }
        for (o = 0; o < sim->chr2.n; ++o) {
            sim->chr2.data[o].irr = 0.0;
        }
    }
    return RCSD_OK;
}

int rcsd_extracellular_junctions(RCSDSim* sim, int* driven, int* dropped) {
    Stimulus* s = &sim->stim[RCSD_STIM_EXTRACELLULAR];
    if (driven) *driven = s->n_junctions;
    if (dropped) *dropped = s->dropped_junctions;
    return RCSD_OK;
}

long rcsd_stimulus_needed(RCSDSim* sim, int mode) {
    if (mode < 0 || mode >= RCSD_STIM_N) {
        return -1;
    }
    return sim->stim[mode].needed;
}

/* whether sample `index` can be read: held, or past the end (zero) */
static int have_sample(const Stimulus* s, long index) {
    if (index < 0) {
        return 1;
    }
    if (index >= s->extent) {
        return 1;
    }
    return index >= s->first && index < s->first + s->n_samples;
}

static const double* sample(const Stimulus* s, long index) {
    if (index < 0 || index >= s->extent) {
        return NULL;
    }
    if (index < s->first || index >= s->first + s->n_samples) {
        return NULL;
    }
    return s->values + (size_t) (index - s->first) * (size_t) s->n_rows;
}

static long index_at(const Stimulus* s, double t) {
    return (long) floor(t / s->stim_dt);
}

/* The samples step `step` needs, mode by mode. With `check` set, only
 * report whether they are held; otherwise inject them. */
static int apply_or_check(RCSDSim* sim, long step, int check, int* need, int* need_mode) {
    const double dt = sim->dt;
    const double t = (double) step * dt;
    const double t_mid = t + 0.5 * dt;
    int m;
    for (m = 0; m < RCSD_STIM_N; ++m) {
        Stimulus* s = &sim->stim[m];
        long i0, i1;
        if (!s->active || s->n_rows == 0) {
            continue;
        }
        switch (m) {
        case RCSD_STIM_CURRENT:
            i0 = index_at(s, t_mid);
            i1 = i0 + 1;
            break;
        case RCSD_STIM_CURRENT_DENSITY:
            i0 = i1 = index_at(s, t);
            break;
        default: /* extracellular and photon flux: read after the update */
            i0 = i1 = index_at(s, t);
            break;
        }
        if (!have_sample(s, i0) || !have_sample(s, i1)) {
            /* a refill starts at i0 so that both samples end up held */
            s->needed = i0;
            if (need) *need = 1;
            if (need_mode) *need_mode = m;
            return RCSD_OK;
        }
        if (check) {
            continue;
        }
#if defined(RCSD_OPENACC)
        /* on the device, behind the kernel that cleared stim_rhs; a sample
         * is an offset into the values, -1 for none (zero) */
        if (m == RCSD_STIM_CURRENT) {
            const double* a = sample(s, i0);
            const double* b = sample(s, i1);
            long oa = a ? (long) (a - s->values) : -1, ob = b ? (long) (b - s->values) : -1;
            double x = t_mid / s->stim_dt - (double) i0;
            int n_rows = s->n_rows, r;
            RCSD_ACC_LOOP(sim)
            for (r = 0; r < n_rows; ++r) {
                const Stimulus* st = &sim->stim[RCSD_STIM_CURRENT];
                double va = oa >= 0 ? st->values[oa + r] : 0.0;
                double vb = ob >= 0 ? st->values[ob + r] : 0.0;
                double amp = (1.0 - x) * va + x * vb;
                int node = st->node[r];
                RCSD_ACC_ATOMIC_UPDATE
                sim->stim_rhs[node] += amp * 1e2 / sim->area[node];
            }
        } else if (m == RCSD_STIM_CURRENT_DENSITY) {
            const double* a = sample(s, i0);
            long oa = a ? (long) (a - s->values) : -1;
            int n_rows = s->n_rows, r;
            if (a) {
                RCSD_ACC_LOOP(sim)
                for (r = 0; r < n_rows; ++r) {
                    const Stimulus* st = &sim->stim[RCSD_STIM_CURRENT_DENSITY];
                    RCSD_ACC_ATOMIC_UPDATE
                    sim->stim_rhs[st->node[r]] += st->values[oa + r];
                }
            }
        }
#else
        if (m == RCSD_STIM_CURRENT) {
            const double* a = sample(s, i0);
            const double* b = sample(s, i1);
            double x = t_mid / s->stim_dt - (double) i0;
            int r;
            for (r = 0; r < s->n_rows; ++r) {
                double va = a ? a[r] : 0.0;
                double vb = b ? b[r] : 0.0;
                double amp = (1.0 - x) * va + x * vb;
                int node = s->node[r];
                sim->stim_rhs[node] += amp * 1e2 / sim->area[node];
            }
        } else if (m == RCSD_STIM_CURRENT_DENSITY) {
            const double* a = sample(s, i0);
            int r;
            if (a) {
                for (r = 0; r < s->n_rows; ++r) {
                    sim->stim_rhs[s->node[r]] += a[r];
                }
            }
        }
#endif
    }
    return RCSD_OK;
}

int rcsd_stimulus_apply(RCSDSim* sim, long step, int* need, int* need_mode) {
    int i;
    if (need != NULL) {
        return apply_or_check(sim, step, 1, need, need_mode);
    }
    /* the field's equivalent currents, set for this step by the previous one;
     * every injected current lands in the IClamp slot of the node */
    RCSD_ACC_LOOP(sim)
    for (i = 0; i < sim->n_nodes; ++i) {
        sim->stim_rhs[i] = 0.0;
        if (sim->ext_amp[i] != 0.0) {
            sim->stim_rhs[i] += sim->ext_amp[i] * 1e2 / sim->area[i];
        }
    }
    return apply_or_check(sim, step, 0, NULL, NULL);
}

#if defined(RCSD_OPENACC)
typedef struct {
    int cell, section, row;
} RowKey;

static int compare_row_key(const void* a, const void* b) {
    const RowKey* x = (const RowKey*) a;
    const RowKey* y = (const RowKey*) b;
    if (x->cell != y->cell) return (x->cell > y->cell) - (x->cell < y->cell);
    if (x->section != y->section) return (x->section > y->section) - (x->section < y->section);
    return (x->row > y->row) - (x->row < y->row);
}

/* the photon-flux row of each opsin and ChR2 segment: the last row naming
 * its cell and section, as the host loop's later rows overwrite earlier
 * ones; -1 for none. Rebuilt by rcsd_run, so the device reads one index a
 * step instead of comparing every row with every opsin */
int rcsd_stimulus_prepare(RCSDSim* sim) {
    const Stimulus* phi = &sim->stim[RCSD_STIM_PHOTON_FLUX];
    RowKey* keys;
    int* opsin_row;
    int* chr2_row;
    int n = phi->n_rows, r;
    size_t o;
    if (!phi->active || n == 0) {
        return RCSD_OK;
    }
    keys = (RowKey*) malloc((size_t) n * sizeof(RowKey));
    opsin_row = (int*) realloc(sim->opsin_row, (sim->opsins.n + 1) * sizeof(int));
    if (opsin_row) sim->opsin_row = opsin_row;
    chr2_row = (int*) realloc(sim->chr2_row, (sim->chr2.n + 1) * sizeof(int));
    if (chr2_row) sim->chr2_row = chr2_row;
    if (!keys || !opsin_row || !chr2_row) {
        free(keys);
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    for (r = 0; r < n; ++r) {
        keys[r].cell = phi->cell[r];
        keys[r].section = phi->section[r];
        keys[r].row = r;
    }
    qsort(keys, (size_t) n, sizeof(RowKey), compare_row_key);
    for (o = 0; o < sim->opsins.n + sim->chr2.n; ++o) {
        int is_opsin = o < sim->opsins.n;
        size_t k = is_opsin ? o : o - sim->opsins.n;
        int cell = is_opsin ? sim->opsins.data[k].cell : sim->chr2.data[k].cell;
        int section = is_opsin ? sim->opsins.data[k].section : sim->chr2.data[k].section;
        /* past the last key at or below (cell, section, n) */
        int lo = 0, hi = n, row = -1;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            RowKey probe = {cell, section, n};
            if (compare_row_key(&keys[mid], &probe) < 0) lo = mid + 1; else hi = mid;
        }
        if (lo > 0 && keys[lo - 1].cell == cell && keys[lo - 1].section == section) {
            row = keys[lo - 1].row;
        }
        if (is_opsin) {
            sim->opsin_row[k] = row;
        } else {
            sim->chr2_row[k] = row;
        }
    }
    free(keys);
    return RCSD_OK;
}

void rcsd_stimulus_after_update(RCSDSim* sim, long step) {
    Stimulus* ext = &sim->stim[RCSD_STIM_EXTRACELLULAR];
    Stimulus* phi = &sim->stim[RCSD_STIM_PHOTON_FLUX];
    /* the callback's current_time is step * dt */
    double t = (double) step * sim->dt;
    if (ext->active && ext->n_rows > 0) {
        const double* col = sample(ext, index_at(ext, t));
        long oc = col ? (long) (col - ext->values) : -1;
        int n_nodes = sim->n_nodes, nj = ext->n_junctions, i, j;
        RCSD_ACC_LOOP(sim)
        for (i = 0; i < n_nodes; ++i) {
            sim->ext_amp[i] = 0.0;
        }
        if (oc >= 0) {
            RCSD_ACC_LOOP(sim)
            for (j = 0; j < nj; ++j) {
                const Stimulus* e = &sim->stim[RCSD_STIM_EXTRACELLULAR];
                const double* c = e->values + oc;
                double delta = (c[e->j_row_b[j]] - c[e->j_row_a[j]]) * e->j_inv_r[j];
                RCSD_ACC_ATOMIC_UPDATE
                sim->ext_amp[e->j_node_a[j]] += delta;
                RCSD_ACC_ATOMIC_UPDATE
                sim->ext_amp[e->j_node_b[j]] -= delta;
            }
        }
    }
    if (phi->active && phi->n_rows > 0) {
        const double* col = sample(phi, index_at(phi, t));
        long oc = col ? (long) (col - phi->values) : -1;
        int n_op = (int) sim->opsins.n, n_ch = (int) sim->chr2.n, o;
        RCSD_ACC_LOOP(sim)
        for (o = 0; o < n_op; ++o) {
            int r = sim->opsin_row[o];
            sim->opsins.data[o].phi =
                (oc >= 0 && r >= 0) ? sim->stim[RCSD_STIM_PHOTON_FLUX].values[oc + r] : 0.0;
        }
        RCSD_ACC_LOOP(sim)
        for (o = 0; o < n_ch; ++o) {
            int r = sim->chr2_row[o];
            sim->chr2.data[o].irr =
                (oc >= 0 && r >= 0) ? sim->stim[RCSD_STIM_PHOTON_FLUX].values[oc + r] : 0.0;
        }
    }
}
#else
void rcsd_stimulus_after_update(RCSDSim* sim, long step) {
    Stimulus* ext = &sim->stim[RCSD_STIM_EXTRACELLULAR];
    Stimulus* phi = &sim->stim[RCSD_STIM_PHOTON_FLUX];
    /* the callback's current_time is step * dt */
    double t = (double) step * sim->dt;
    if (ext->active && ext->n_rows > 0) {
        const double* col = sample(ext, index_at(ext, t));
        int i;
        RCSD_ACC_LOOP(sim)
        for (i = 0; i < sim->n_nodes; ++i) {
            sim->ext_amp[i] = 0.0;
        }
        if (col != NULL) {
            int j;
            for (j = 0; j < ext->n_junctions; ++j) {
                double delta = (col[ext->j_row_b[j]] - col[ext->j_row_a[j]]) * ext->j_inv_r[j];
                sim->ext_amp[ext->j_node_a[j]] += delta;
                sim->ext_amp[ext->j_node_b[j]] -= delta;
            }
        }
    }
    if (phi->active && phi->n_rows > 0) {
        const double* col = sample(phi, index_at(phi, t));
        size_t o;
        for (o = 0; o < sim->opsins.n; ++o) {
            sim->opsins.data[o].phi = 0.0;
        }
        for (o = 0; o < sim->chr2.n; ++o) {
            sim->chr2.data[o].irr = 0.0;
        }
        if (col != NULL) {
            int r;
            for (r = 0; r < phi->n_rows; ++r) {
                for (o = 0; o < sim->opsins.n; ++o) {
                    Opsin* op = &sim->opsins.data[o];
                    if (op->cell == phi->cell[r] && op->section == phi->section[r]) {
                        op->phi = col[r];
                    }
                }
                /* every segment of the section: the light is the section's */
                for (o = 0; o < sim->chr2.n; ++o) {
                    ChR2* ch = &sim->chr2.data[o];
                    if (ch->cell == phi->cell[r] && ch->section == phi->section[r]) {
                        ch->irr = col[r];
                    }
                }
            }
        }
    }
}
#endif
