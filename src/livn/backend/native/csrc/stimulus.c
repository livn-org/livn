/* Stimulus delivery, in the four modes the Python layer hands over.
 *
 * The timing mirrors the NEURON backend: an extracellular field is turned
 * into equivalent axial currents by a per-step callback that runs after the
 * voltage update, so the field sampled at step s drives step s + 1; a
 * photon flux set by the same callback enters the opsin kinetics of step s;
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

static int find_row(const Stimulus* s, int cell, int section) {
    int r;
    for (r = 0; r < s->n_rows; ++r) {
        if (s->cell[r] == cell && s->section[r] == section) {
            return r;
        }
    }
    return -1;
}

/* the junctions of every cell the field reaches on both sides */
static int build_junctions(RCSDSim* sim, Stimulus* s) {
    size_t sec_index;
    int cap = 0, n = 0, dropped = 0;
    free(s->j_row_a);
    free(s->j_row_b);
    free(s->j_node_a);
    free(s->j_node_b);
    free(s->j_inv_r);
    s->j_row_a = s->j_row_b = s->j_node_a = s->j_node_b = NULL;
    s->j_inv_r = NULL;
    for (sec_index = 0; sec_index < sim->sections.n; ++sec_index) {
        const Section* child = &sim->sections.data[sec_index];
        const Section* parent;
        const Cell* cell;
        int row_a, row_b, node_a, node_b, j;
        double resistance;
        if (child->parent_section < 0) {
            continue;
        }
        parent = &sim->sections.data[child->parent_section];
        cell = &sim->cells.data[child->cell];
        resistance = half_axial(child) + half_axial(parent);
        if (resistance <= 0.0) {
            continue;
        }
        row_a = find_row(s, child->cell, (int) sec_index - cell->sec0);
        row_b = find_row(s, child->cell, child->parent_section - cell->sec0);
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
        for (i = 0; i < sim->n_nodes; ++i) {
            sim->ext_amp[i] = 0.0;
        }
    }
    if (mode == RCSD_STIM_PHOTON_FLUX) {
        size_t o;
        for (o = 0; o < sim->opsins.n; ++o) {
            sim->opsins.data[o].phi = 0.0;
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
                sim->rhs[node] += amp * 1e2 / sim->area[node];
            }
        } else if (m == RCSD_STIM_CURRENT_DENSITY) {
            const double* a = sample(s, i0);
            int r;
            if (a) {
                for (r = 0; r < s->n_rows; ++r) {
                    sim->rhs[s->node[r]] += a[r];
                }
            }
        }
    }
    return RCSD_OK;
}

int rcsd_stimulus_apply(RCSDSim* sim, long step, int* need, int* need_mode) {
    int i;
    if (need != NULL) {
        return apply_or_check(sim, step, 1, need, need_mode);
    }
    /* the field's equivalent currents, set for this step by the previous one */
    for (i = 0; i < sim->n_nodes; ++i) {
        if (sim->ext_amp[i] != 0.0) {
            sim->rhs[i] += sim->ext_amp[i] * 1e2 / sim->area[i];
        }
    }
    return apply_or_check(sim, step, 0, NULL, NULL);
}

void rcsd_stimulus_after_update(RCSDSim* sim, long step) {
    Stimulus* ext = &sim->stim[RCSD_STIM_EXTRACELLULAR];
    Stimulus* phi = &sim->stim[RCSD_STIM_PHOTON_FLUX];
    /* the callback's current_time is step * dt */
    double t = (double) step * sim->dt;
    if (ext->active && ext->n_rows > 0) {
        const double* col = sample(ext, index_at(ext, t));
        int i;
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
        if (col != NULL) {
            int r;
            for (r = 0; r < phi->n_rows; ++r) {
                for (o = 0; o < sim->opsins.n; ++o) {
                    Opsin* op = &sim->opsins.data[o];
                    if (op->cell == phi->cell[r] && op->section == phi->section[r]) {
                        op->phi = col[r];
                    }
                }
            }
        }
    }
}
