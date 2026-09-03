/* Cells, geometry, initialisation and the timestep. */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "mech.h"

#ifdef RCSD_PROFILE
#include <time.h>
static double prof_acc[10];
static const char* prof_name[10] = {"events", "noise", "membrane", "syn_currents", "noise+opsin+stim", "matrix+solve", "update", "states", "syn_states", "spikes"};
static double prof_now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
#define PROF_MARK(i) do { double now_ = prof_now(); prof_acc[i] += now_ - prof_t; prof_t = now_; } while (0)
void rcsd_profile_dump(void) { int i; double total = 0; for (i = 0; i < 10; ++i) total += prof_acc[i]; for (i = 0; i < 10; ++i) fprintf(stderr, "%-18s %8.2f s %5.1f%%\n", prof_name[i], prof_acc[i], 100 * prof_acc[i] / total); }
#else
#define PROF_MARK(i)
#endif

static char g_error[512];

void rcsd_set_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof g_error, fmt, ap);
    va_end(ap);
}

const char* rcsd_version(void) {
    return RCSD_VERSION;
}

const char* rcsd_last_error(void) {
    return g_error;
}

int dyn_reserve(void** data, size_t* cap, size_t n, size_t itemsize) {
    size_t want = *cap ? *cap : 16;
    void* grown;
    if (n <= *cap) {
        return 1;
    }
    while (want < n) {
        want *= 2;
    }
    grown = realloc(*data, want * itemsize);
    if (grown == NULL) {
        rcsd_set_error("out of memory");
        return 0;
    }
    *data = grown;
    *cap = want;
    return 1;
}

/* ------------------------------------------------------------------------- */
/* lifecycle                                                                  */
/* ------------------------------------------------------------------------- */

RCSDSim* rcsd_create(double celsius, double v_init) {
    RCSDSim* sim = (RCSDSim*) calloc(1, sizeof(RCSDSim));
    int m;
    if (sim == NULL) {
        rcsd_set_error("out of memory");
        return NULL;
    }
    sim->celsius = celsius;
    sim->v_init = v_init;
    sim->dt = 0.025;
    sim->cj = 2.0 / sim->dt;
    sim->geometry_dirty = 1;
    for (m = 0; m < RCSD_STIM_N; ++m) {
        sim->stim[m].needed = -1;
    }
    return sim;
}

static void free_traces(Trace* traces, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        free(traces[i].values.data);
    }
}

void rcsd_destroy(RCSDSim* sim) {
    size_t i;
    int m;
    if (sim == NULL) {
        return;
    }
    DYN_FREE(sim->cells);
    DYN_FREE(sim->sections);
    free(sim->parent);
    free(sim->section_of);
    free(sim->is_centre);
    free(sim->area);
    free(sim->rinv);
    free(sim->coef_a);
    free(sim->coef_b);
    free(sim->d);
    free(sim->rhs);
    free(sim->sav_rhs);
    free(sim->sav_d);
    free(sim->v);
    free(sim->state);
    free(sim->param);
    free(sim->mech);
    free(sim->dinadv);
    free(sim->dikdv);
    free(sim->dicadv);
    free(sim->ext_amp);
    free(sim->stim_amp);
    free(sim->stim_dens);
    DYN_FREE(sim->synapses);
    free(sim->sp);
    free(sim->ss);
    free(sim->sc);
    DYN_FREE(sim->connections);
    free(sim->w);
    for (i = 0; i < sim->inputs.n; ++i) {
        free(sim->inputs.data[i].times);
    }
    DYN_FREE(sim->inputs);
    free(sim->out_start);
    free(sim->out_conn);
    free(sim->in_start);
    free(sim->in_conn);
    if (sim->buckets) {
        for (i = 0; i < (size_t) sim->n_slots; ++i) {
            free(sim->buckets[i].data);
        }
        free(sim->buckets);
    }
    DYN_FREE(sim->noise);
    DYN_FREE(sim->opsins);
    for (m = 0; m < RCSD_STIM_N; ++m) {
        rcsd_stimulus_free(&sim->stim[m]);
    }
    DYN_FREE(sim->spike_cells);
    DYN_FREE(sim->spike_times);
    free(sim->record_spikes);
    free_traces(sim->v_traces.data, sim->v_traces.n);
    DYN_FREE(sim->v_traces);
    free_traces(sim->i_traces.data, sim->i_traces.n);
    DYN_FREE(sim->i_traces);
    free(sim);
}

/* ------------------------------------------------------------------------- */
/* nodes                                                                      */
/* ------------------------------------------------------------------------- */

#define GROW(ptr, type, old, new_cap)                                    \
    do {                                                                 \
        type* grown = (type*) realloc((ptr), (new_cap) * sizeof(type));  \
        if (grown == NULL) {                                             \
            rcsd_set_error("out of memory");                             \
            return RCSD_ERROR;                                           \
        }                                                                \
        memset(grown + (old), 0, ((new_cap) - (old)) * sizeof(type));    \
        (ptr) = grown;                                                   \
    } while (0)

int rcsd_alloc_nodes(RCSDSim* sim, int n) {
    int want = sim->n_nodes + n;
    int cap;
    if (want <= sim->cap_nodes) {
        return RCSD_OK;
    }
    cap = sim->cap_nodes ? sim->cap_nodes : 64;
    while (cap < want) {
        cap *= 2;
    }
    GROW(sim->parent, int, sim->cap_nodes, cap);
    GROW(sim->section_of, int, sim->cap_nodes, cap);
    GROW(sim->is_centre, int, sim->cap_nodes, cap);
    GROW(sim->area, double, sim->cap_nodes, cap);
    GROW(sim->rinv, double, sim->cap_nodes, cap);
    GROW(sim->coef_a, double, sim->cap_nodes, cap);
    GROW(sim->coef_b, double, sim->cap_nodes, cap);
    GROW(sim->d, double, sim->cap_nodes, cap);
    GROW(sim->rhs, double, sim->cap_nodes, cap);
    GROW(sim->sav_rhs, double, sim->cap_nodes, cap);
    GROW(sim->sav_d, double, sim->cap_nodes, cap);
    GROW(sim->v, double, sim->cap_nodes, cap);
    GROW(sim->state, double, (size_t) sim->cap_nodes * RCSD_NSTATE, (size_t) cap * RCSD_NSTATE);
    GROW(sim->param, double, (size_t) sim->cap_nodes * RCSD_NPARAM, (size_t) cap * RCSD_NPARAM);
    GROW(sim->mech, unsigned, sim->cap_nodes, cap);
    GROW(sim->dinadv, double, sim->cap_nodes, cap);
    GROW(sim->dikdv, double, sim->cap_nodes, cap);
    GROW(sim->dicadv, double, sim->cap_nodes, cap);
    GROW(sim->ext_amp, double, sim->cap_nodes, cap);
    GROW(sim->stim_amp, double, sim->cap_nodes, cap);
    GROW(sim->stim_dens, double, sim->cap_nodes, cap);
    sim->cap_nodes = cap;
    return RCSD_OK;
}

static int new_node(RCSDSim* sim, int parent, int section, int centre) {
    int i;
    if (rcsd_alloc_nodes(sim, 1) != RCSD_OK) {
        return RCSD_ERROR;
    }
    i = sim->n_nodes++;
    sim->parent[i] = parent;
    sim->section_of[i] = section;
    sim->is_centre[i] = centre;
    sim->area[i] = 100.0;
    sim->mech[i] = 0;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_CAO] = 2.0;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_CAI0] = 1e-5;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_KD_KCA] = 0.0005;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_D_NA] = 0.2;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_BETA_NA] = 0.075;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_NAI0] = 15.0;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_NAO0] = 145.0;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_D_K] = 0.2;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_BETA_K] = 0.075;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_KI0] = 145.0;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_KO0] = 5.0;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_F_CA] = 0.004;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_KCA_CA] = 8.0;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_ALPHA_CA] = 1.0;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_VHALF_NAS] = -35.0;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_SLOPE_NAS] = 7.8;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_E_PAS] = -70.0;
    sim->param[(size_t) i * RCSD_NPARAM + RCSD_P_CM] = 1.0;
    return i;
}

/* ------------------------------------------------------------------------- */
/* cells and sections                                                         */
/* ------------------------------------------------------------------------- */

int rcsd_cell_count(RCSDSim* sim) {
    return (int) sim->cells.n;
}

int rcsd_node_count(RCSDSim* sim) {
    return sim->n_nodes;
}

int rcsd_add_cell(RCSDSim* sim, int gid, int population, double v_threshold, double v_hold,
                  double tref) {
    Cell cell;
    memset(&cell, 0, sizeof cell);
    cell.gid = gid;
    cell.population = population;
    cell.sec0 = (int) sim->sections.n;
    cell.node0 = sim->n_nodes;
    cell.root_node = -1;
    cell.soma_section = -1;
    cell.soma_node = -1;
    cell.v_threshold = v_threshold;
    cell.v_hold = v_hold;
    cell.tref = tref;
    cell.t_last_spike = -1e9;
    cell.opsin = -1;
    DYN_PUSH(sim->cells, cell);
    sim->wired = 0;
    return (int) sim->cells.n - 1;
}

int rcsd_cell_set(RCSDSim* sim, int cell, double v_threshold, double v_hold, double tref) {
    Cell* c;
    if (cell < 0 || (size_t) cell >= sim->cells.n) {
        rcsd_set_error("no cell %d", cell);
        return RCSD_ERROR;
    }
    c = &sim->cells.data[cell];
    c->v_threshold = v_threshold;
    c->v_hold = v_hold;
    c->tref = tref;
    return RCSD_OK;
}

static Section* section_at(RCSDSim* sim, int section) {
    if (section < 0 || (size_t) section >= sim->sections.n) {
        rcsd_set_error("no section %d", section);
        return NULL;
    }
    return &sim->sections.data[section];
}

/* NEURON's node_exact: which node a position on a section refers to */
static int node_exact(RCSDSim* sim, const Section* sec, double x) {
    int j;
    (void) sim;
    if (x >= 1.0) {
        return sec->end_node;
    }
    if (x <= 0.0) {
        return sec->parent_node;
    }
    j = (int) (x * sec->nseg);
    if (j >= sec->nseg) {
        j = sec->nseg - 1;
    }
    return sec->node0 + j;
}

int rcsd_section_node(RCSDSim* sim, int section, double x) {
    /* livn's segment_at: the centre of the segment containing x */
    Section* sec = section_at(sim, section);
    int j;
    if (sec == NULL) {
        return RCSD_ERROR;
    }
    j = (int) (x * sec->nseg);
    if (j < 0) {
        j = 0;
    }
    if (j >= sec->nseg) {
        j = sec->nseg - 1;
    }
    return sec->node0 + j;
}

int rcsd_add_section(RCSDSim* sim, int cell, int kind, int nseg, double L, double diam,
                     double Ra, double cm, unsigned mechanisms, int parent_section,
                     double parent_x) {
    Section sec;
    Cell* c;
    int j, parent_node, prev, index;
    size_t p;
    if (cell < 0 || (size_t) cell >= sim->cells.n) {
        rcsd_set_error("no cell %d", cell);
        return RCSD_ERROR;
    }
    if (nseg < 1) {
        rcsd_set_error("nseg must be >= 1, got %d", nseg);
        return RCSD_ERROR;
    }
    c = &sim->cells.data[cell];
    if ((size_t) cell != sim->cells.n - 1) {
        rcsd_set_error("sections have to be added cell by cell");
        return RCSD_ERROR;
    }
    memset(&sec, 0, sizeof sec);
    sec.cell = cell;
    sec.kind = kind;
    sec.nseg = nseg;
    sec.L = L;
    sec.diam = diam;
    sec.Ra = Ra;
    sec.mech = mechanisms;
    sec.parent_section = parent_section;
    sec.parent_x = parent_x;
    index = (int) sim->sections.n;

    if (parent_section < 0) {
        if (c->root_node >= 0) {
            rcsd_set_error("cell %d already has a root section", cell);
            return RCSD_ERROR;
        }
        parent_node = new_node(sim, -1, -1, 0);
        if (parent_node < 0) {
            return RCSD_ERROR;
        }
        c->root_node = parent_node;
    } else {
        Section* parent = section_at(sim, parent_section);
        if (parent == NULL || parent->cell != cell) {
            rcsd_set_error("parent section %d does not belong to cell %d", parent_section,
                           cell);
            return RCSD_ERROR;
        }
        parent_node = node_exact(sim, parent, parent_x);
    }
    sec.parent_node = parent_node;

    prev = parent_node;
    sec.node0 = sim->n_nodes;
    for (j = 0; j < nseg; ++j) {
        int node = new_node(sim, prev, index, 1);
        if (node < 0) {
            return RCSD_ERROR;
        }
        sim->mech[node] = mechanisms;
        sim->param[(size_t) node * RCSD_NPARAM + RCSD_P_CM] = cm;
        prev = node;
    }
    sec.end_node = new_node(sim, prev, index, 0);
    if (sec.end_node < 0) {
        return RCSD_ERROR;
    }
    p = sim->sections.n;
    DYN_PUSH(sim->sections, sec);
    (void) p;
    c->nsec += 1;
    c->nnode = sim->n_nodes - c->node0;
    if (kind == RCSD_SEC_SOMA && c->soma_section < 0) {
        c->soma_section = index;
        c->soma_node = rcsd_section_node(sim, index, 0.5);
    }
    if (c->soma_section < 0) {
        /* until a soma turns up, detect at the first section */
        c->soma_node = rcsd_section_node(sim, c->sec0, 0.5);
    }
    sim->geometry_dirty = 1;
    return index;
}

int rcsd_cell_section_count(RCSDSim* sim, int cell) {
    if (cell < 0 || (size_t) cell >= sim->cells.n) {
        return RCSD_ERROR;
    }
    return sim->cells.data[cell].nsec;
}

int rcsd_cell_section(RCSDSim* sim, int cell, int index) {
    Cell* c;
    if (cell < 0 || (size_t) cell >= sim->cells.n) {
        return RCSD_ERROR;
    }
    c = &sim->cells.data[cell];
    if (index < 0 || index >= c->nsec) {
        return RCSD_ERROR;
    }
    return c->sec0 + index;
}

int rcsd_section_set(RCSDSim* sim, int section, int param, double value) {
    Section* sec = section_at(sim, section);
    int j;
    if (sec == NULL) {
        return RCSD_ERROR;
    }
    if (param < 0 || param >= RCSD_NPARAM) {
        rcsd_set_error("no parameter %d", param);
        return RCSD_ERROR;
    }
    for (j = 0; j < sec->nseg; ++j) {
        sim->param[(size_t) (sec->node0 + j) * RCSD_NPARAM + param] = value;
    }
    return RCSD_OK;
}

double rcsd_section_get(RCSDSim* sim, int section, int param) {
    Section* sec = section_at(sim, section);
    int node;
    if (sec == NULL || param < 0 || param >= RCSD_NPARAM) {
        return NAN;
    }
    node = rcsd_section_node(sim, section, 0.5);
    return sim->param[(size_t) node * RCSD_NPARAM + param];
}

int rcsd_section_geometry(RCSDSim* sim, int section, double L, double diam, double Ra) {
    Section* sec = section_at(sim, section);
    if (sec == NULL) {
        return RCSD_ERROR;
    }
    sec->L = L;
    sec->diam = diam;
    sec->Ra = Ra;
    sim->geometry_dirty = 1;
    return RCSD_OK;
}

int rcsd_section_info(RCSDSim* sim, int section, int* nseg, double* L, double* diam,
                      double* Ra, unsigned* mechanisms) {
    Section* sec = section_at(sim, section);
    if (sec == NULL) {
        return RCSD_ERROR;
    }
    if (nseg) *nseg = sec->nseg;
    if (L) *L = sec->L;
    if (diam) *diam = sec->diam;
    if (Ra) *Ra = sec->Ra;
    if (mechanisms) *mechanisms = sec->mech;
    return RCSD_OK;
}

double rcsd_node_state(RCSDSim* sim, int node, int state) {
    if (node < 0 || node >= sim->n_nodes) {
        return NAN;
    }
    if (state == RCSD_S_V) {
        return sim->v[node];
    }
    if (state < 0 || state >= RCSD_NSTATE) {
        return NAN;
    }
    return sim->state[(size_t) node * RCSD_NSTATE + state];
}

double rcsd_node_area(RCSDSim* sim, int node) {
    if (node < 0 || node >= sim->n_nodes) {
        return NAN;
    }
    return sim->area[node];
}

/* NEURON's nrn_area_ri for a stylised section, plus the coupling coefficients */
int rcsd_build_geometry(RCSDSim* sim) {
    size_t s;
    int i;
    for (s = 0; s < sim->sections.n; ++s) {
        Section* sec = &sim->sections.data[s];
        double dx = sec->L / (double) sec->nseg;
        double rright = 0.0;
        double rleft;
        int j;
        for (j = 0; j < sec->nseg; ++j) {
            int node = sec->node0 + j;
            sim->area[node] = M_PI * dx * sec->diam;
            rleft = 1e-2 * sec->Ra * (dx / 2.0) / (M_PI * sec->diam * sec->diam / 4.0);
            sim->rinv[node] = 1.0 / (rleft + rright);
            rright = rleft;
        }
        sim->area[sec->end_node] = 1e2;
        sim->rinv[sec->end_node] = 1.0 / rright;
    }
    for (i = 0; i < sim->n_nodes; ++i) {
        int p = sim->parent[i];
        if (p < 0) {
            sim->area[i] = 1e2;
            sim->coef_a[i] = 0.0;
            sim->coef_b[i] = 0.0;
            continue;
        }
        sim->coef_b[i] = -1e2 * sim->rinv[i] / sim->area[i];
        sim->coef_a[i] = -1e2 * sim->rinv[i] / sim->area[p];
    }
    sim->geometry_dirty = 0;
    return RCSD_OK;
}

/* ------------------------------------------------------------------------- */
/* membrane mechanisms                                                        */
/* ------------------------------------------------------------------------- */

#define ST(node, s) sim->state[(size_t) (node) * RCSD_NSTATE + (s)]
#define PR(node, p) sim->param[(size_t) (node) * RCSD_NPARAM + (p)]

/* Every density mechanism's BREAKPOINT at the present states, with the
 * numerical conductance NEURON derives from `_nrn_current(v + .001)`.
 * Fills rhs, d, the ion totals and their derivatives, i_pas and sav_*. */
static void eval_membrane(RCSDSim* sim) {
    const double celsius = sim->celsius;
    const double fN = can_f(celsius);
    const double fL = cal_f(celsius);
    int i;
    for (i = 0; i < sim->n_nodes; ++i) {
        unsigned mech = sim->mech[i];
        double v = sim->v[i];
        double vp = v + 0.001;
        double rhs = 0.0, dd = 0.0;
        double ina = 0.0, ik = 0.0, ica = 0.0, ipas = 0.0;
        double dina = 0.0, dik = 0.0, dica = 0.0;
        double ena = 0.0, ek = 0.0;
        double cai, cao;

        sim->rhs[i] = 0.0;
        sim->d[i] = 0.0;
        sim->sav_rhs[i] = 0.0;
        sim->sav_d[i] = 0.0;
        sim->dinadv[i] = 0.0;
        sim->dikdv[i] = 0.0;
        sim->dicadv[i] = 0.0;
        if (!sim->is_centre[i]) {
            continue;
        }
        if (mech & RCSD_M_NA_CONC) {
            ena = nernst(ST(i, RCSD_S_NAI), ST(i, RCSD_S_NAO), 1.0, celsius, RCSD_GASCONSTANT,
                         RCSD_FARADAY);
        }
        if (mech & RCSD_M_K_CONC) {
            ek = nernst(ST(i, RCSD_S_KI), ST(i, RCSD_S_KO), 1.0, celsius, RCSD_GASCONSTANT,
                        RCSD_FARADAY);
        }
        ST(i, RCSD_S_ENA) = ena;
        ST(i, RCSD_S_EK) = ek;
        cai = ST(i, RCSD_S_CAI);
        cao = PR(i, RCSD_P_CAO);

        if (mech & RCSD_M_PAS) {
            double g = PR(i, RCSD_P_G_PAS), e = PR(i, RCSD_P_E_PAS);
            double i0 = g * (v - e), i1 = g * (vp - e);
            ipas = i0;
            rhs -= i0;
            dd += (i1 - i0) / 0.001;
        }
        if (mech & RCSD_M_CONSTANT) {
            rhs -= PR(i, RCSD_P_IC);
        }
        if (mech & RCSD_M_NAS) {
            double gmax = PR(i, RCSD_P_GMAX_NAS);
            double minf = nas_minf(v, PR(i, RCSD_P_VHALF_NAS), PR(i, RCSD_P_SLOPE_NAS));
            double h = ST(i, RCSD_S_H);
            double i1 = nas_current(vp, gmax, minf, h, ena);
            double i0 = nas_current(v, gmax, minf, h, ena);
            double g = (i1 - i0) / 0.001;
            ina += i0;
            dina += g;
            rhs -= i0;
            dd += g;
        }
        if (mech & RCSD_M_KDR) {
            double gmax = PR(i, RCSD_P_GMAX_KDR);
            double n = ST(i, RCSD_S_N);
            double i1 = kdr_current(vp, gmax, n, ek);
            double i0 = kdr_current(v, gmax, n, ek);
            double g = (i1 - i0) / 0.001;
            ik += i0;
            dik += g;
            rhs -= i0;
            dd += g;
        }
        if (mech & RCSD_M_CAN) {
            double gmax = PR(i, RCSD_P_GMAX_CAN);
            double m = ST(i, RCSD_S_MN), h = ST(i, RCSD_S_HN);
            double i1 = can_current(vp, gmax, m, h, cai, cao, fN);
            double i0 = can_current(v, gmax, m, h, cai, cao, fN);
            double g = (i1 - i0) / 0.001;
            ica += i0;
            dica += g;
            rhs -= i0;
            dd += g;
        }
        if (mech & RCSD_M_CAL) {
            double gmax = PR(i, RCSD_P_GMAX_CAL);
            double m = ST(i, RCSD_S_ML);
            double i1 = cal_current(vp, gmax, m, cai, cao, fL);
            double i0 = cal_current(v, gmax, m, cai, cao, fL);
            double g = (i1 - i0) / 0.001;
            ica += i0;
            dica += g;
            rhs -= i0;
            dd += g;
        }
        if (mech & RCSD_M_KCA) {
            double gmax = PR(i, RCSD_P_GMAX_KCA), Kd = PR(i, RCSD_P_KD_KCA);
            double i1 = kca_current(vp, gmax, Kd, cai, ek);
            double i0 = kca_current(v, gmax, Kd, cai, ek);
            double g = (i1 - i0) / 0.001;
            ik += i0;
            dik += g;
            rhs -= i0;
            dd += g;
        }
        if (mech & RCSD_M_KA_V1IN) {
            double gmax = PR(i, RCSD_P_GMAX_KA);
            double a = ST(i, RCSD_S_A), b = ST(i, RCSD_S_B);
            double i1 = ka_current(vp, gmax, a, b, ek);
            double i0 = ka_current(v, gmax, a, b, ek);
            double g = (i1 - i0) / 0.001;
            ik += i0;
            dik += g;
            rhs -= i0;
            dd += g;
        }
        ST(i, RCSD_S_INA) = ina;
        ST(i, RCSD_S_IK) = ik;
        ST(i, RCSD_S_ICA) = ica;
        ST(i, RCSD_S_IPAS) = ipas;
        sim->dinadv[i] = dina;
        sim->dikdv[i] = dik;
        sim->dicadv[i] = dica;
        sim->rhs[i] = rhs;
        sim->d[i] = dd;
    }
}

/* the cnexp state updates (nrn_state) at the new voltage */
static void membrane_states(RCSDSim* sim) {
    const double dt = sim->dt;
    int i;
    for (i = 0; i < sim->n_nodes; ++i) {
        unsigned mech = sim->mech[i];
        double v = sim->v[i];
        if (!sim->is_centre[i]) {
            continue;
        }
        if (mech & RCSD_M_NAS) {
            ST(i, RCSD_S_H) = cnexp_relax(ST(i, RCSD_S_H), nas_hinf(v), nas_htau(v), dt);
        }
        if (mech & RCSD_M_KDR) {
            ST(i, RCSD_S_N) = cnexp_relax(ST(i, RCSD_S_N), kdr_ninf(v), kdr_ntau(v), dt);
        }
        if (mech & RCSD_M_CAN) {
            ST(i, RCSD_S_MN) = cnexp_relax(ST(i, RCSD_S_MN), can_minf(v), CAN_MTAU, dt);
            ST(i, RCSD_S_HN) = cnexp_relax(ST(i, RCSD_S_HN), can_hinf(v), CAN_HTAU, dt);
        }
        if (mech & RCSD_M_CAL) {
            ST(i, RCSD_S_ML) = cnexp_relax(ST(i, RCSD_S_ML), cal_minf(v), CAL_MTAU, dt);
        }
        if (mech & RCSD_M_KA_V1IN) {
            ST(i, RCSD_S_A) = cnexp_relax(ST(i, RCSD_S_A), ka_ainf(v), KA_ATAU, dt);
            ST(i, RCSD_S_B) = cnexp_relax(ST(i, RCSD_S_B), ka_binf(v), KA_BTAU, dt);
        }
        if (mech & RCSD_M_NA_CONC) {
            double ina = ST(i, RCSD_S_INA);
            double d = PR(i, RCSD_P_D_NA), beta = PR(i, RCSD_P_BETA_NA);
            ST(i, RCSD_S_NAI) = conc_step(ST(i, RCSD_S_NAI), PR(i, RCSD_P_NAI0), -ina, beta, d,
                                          dt, RCSD_FARADAY);
            ST(i, RCSD_S_NAO) = conc_step(ST(i, RCSD_S_NAO), PR(i, RCSD_P_NAO0), ina, beta, d,
                                          dt, RCSD_FARADAY);
        }
        if (mech & RCSD_M_K_CONC) {
            double ik = ST(i, RCSD_S_IK);
            double d = PR(i, RCSD_P_D_K), beta = PR(i, RCSD_P_BETA_K);
            ST(i, RCSD_S_KI) = conc_step(ST(i, RCSD_S_KI), PR(i, RCSD_P_KI0), -ik, beta, d, dt,
                                         RCSD_FARADAY);
            ST(i, RCSD_S_KO) = conc_step(ST(i, RCSD_S_KO), PR(i, RCSD_P_KO0), ik, beta, d, dt,
                                         RCSD_FARADAY);
        }
        if (mech & RCSD_M_CA_CONC) {
            ST(i, RCSD_S_CAI) = ca_conc_step(ST(i, RCSD_S_CAI), PR(i, RCSD_P_CAI0),
                                             ST(i, RCSD_S_ICA), ST(i, RCSD_S_IREST),
                                             PR(i, RCSD_P_F_CA), PR(i, RCSD_P_ALPHA_CA),
                                             PR(i, RCSD_P_KCA_CA), dt);
        }
    }
}

/* finitialize: every state at its steady value for v0 */
static void init_states(RCSDSim* sim, int cell_index, double v0) {
    Cell* c = &sim->cells.data[cell_index];
    int i;
    for (i = c->node0; i < c->node0 + c->nnode; ++i) {
        unsigned mech = sim->mech[i];
        sim->v[i] = v0;
        if (!sim->is_centre[i]) {
            continue;
        }
        ST(i, RCSD_S_H) = nas_hinf(v0);
        ST(i, RCSD_S_N) = kdr_ninf(v0);
        ST(i, RCSD_S_MN) = can_minf(v0);
        ST(i, RCSD_S_HN) = can_hinf(v0);
        ST(i, RCSD_S_ML) = cal_minf(v0);
        ST(i, RCSD_S_A) = ka_ainf(v0);
        ST(i, RCSD_S_B) = ka_binf(v0);
        ST(i, RCSD_S_CAI) = PR(i, RCSD_P_CAI0);
        ST(i, RCSD_S_NAI) = PR(i, RCSD_P_NAI0);
        ST(i, RCSD_S_NAO) = PR(i, RCSD_P_NAO0);
        ST(i, RCSD_S_KI) = PR(i, RCSD_P_KI0);
        ST(i, RCSD_S_KO) = PR(i, RCSD_P_KO0);
        (void) mech;
    }
}

static void pin_cell(RCSDSim* sim, int cell_index) {
    Cell* c = &sim->cells.data[cell_index];
    int i;
    for (i = c->node0; i < c->node0 + c->nnode; ++i) {
        if (!sim->is_centre[i] || !(sim->mech[i] & RCSD_M_CONSTANT)) {
            continue;
        }
        PR(i, RCSD_P_IC) = -(ST(i, RCSD_S_INA) + ST(i, RCSD_S_IK) + ST(i, RCSD_S_ICA) +
                             ST(i, RCSD_S_IPAS));
    }
}

/* ------------------------------------------------------------------------- */
/* initialisation                                                             */
/* ------------------------------------------------------------------------- */

int rcsd_set_dt(RCSDSim* sim, double dt) {
    if (!(dt > 0.0)) {
        rcsd_set_error("dt must be positive, got %g", dt);
        return RCSD_ERROR;
    }
    if (sim->initialized && fabs(dt - sim->dt) > 1e-12) {
        rcsd_set_error("dt cannot change once the simulation is initialised");
        return RCSD_ERROR;
    }
    sim->dt = dt;
    sim->cj = 2.0 / dt;
    {
        size_t i;
        for (i = 0; i < sim->synapses.n; ++i) {
            rcsd_synapse_finalize_factor(sim, (int) i);
        }
    }
    return RCSD_OK;
}

double rcsd_dt(RCSDSim* sim) {
    return sim->dt;
}

int rcsd_set_v_init(RCSDSim* sim, double v_init) {
    sim->v_init = v_init;
    return RCSD_OK;
}

int rcsd_initialized(RCSDSim* sim) {
    return sim->initialized;
}

long rcsd_step(RCSDSim* sim) {
    return sim->step;
}

double rcsd_time(RCSDSim* sim) {
    return sim->t;
}

static void reset_traces(Trace* traces, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) {
        traces[i].next = 0;
        traces[i].values.n = 0;
    }
}

int rcsd_init(RCSDSim* sim) {
    size_t c;
    int i;
    if (sim->cells.n == 0) {
        rcsd_set_error("nothing to initialise: no cells");
        return RCSD_ERROR;
    }
    if (sim->geometry_dirty) {
        rcsd_build_geometry(sim);
    }
    if (rcsd_wire(sim) != RCSD_OK) {
        return RCSD_ERROR;
    }
    for (i = 0; i < sim->n_nodes; ++i) {
        sim->ext_amp[i] = 0.0;
        sim->stim_amp[i] = 0.0;
        sim->stim_dens[i] = 0.0;
    }

    /* resting.pin(): one initialisation per cell at its hold potential */
    for (c = 0; c < sim->cells.n; ++c) {
        init_states(sim, (int) c, sim->cells.data[c].v_hold);
    }
    eval_membrane(sim);
    for (c = 0; c < sim->cells.n; ++c) {
        pin_cell(sim, (int) c);
    }

    /* then finitialize(v_init), where Ca_conc captures irest from the
     * currents of the initialisation before it */
    for (c = 0; c < sim->cells.n; ++c) {
        init_states(sim, (int) c, sim->v_init);
    }
    eval_membrane(sim);
    for (i = 0; i < sim->n_nodes; ++i) {
        if (sim->is_centre[i] && (sim->mech[i] & RCSD_M_CA_CONC)) {
            ST(i, RCSD_S_IREST) = ST(i, RCSD_S_ICA);
        }
        ST(i, RCSD_S_IMEM) = 0.0;
    }
    for (c = 0; c < sim->cells.n; ++c) {
        Cell* cell = &sim->cells.data[c];
        cell->above = 0;
        cell->t_last_spike = -1e9;
    }
    for (c = 0; c < sim->inputs.n; ++c) {
        sim->inputs.data[c].cursor = 0;
    }
    rcsd_synapse_init_states(sim);
    rcsd_noise_init(sim);
    rcsd_opsin_init(sim);
    rcsd_events_clear(sim);
    sim->step = 0;
    sim->t = 0.0;
    sim->spike_cells.n = 0;
    sim->spike_times.n = 0;
    reset_traces(sim->v_traces.data, sim->v_traces.n);
    reset_traces(sim->i_traces.data, sim->i_traces.n);
    sim->initialized = 1;
    return RCSD_OK;
}

int rcsd_pin_resting(RCSDSim* sim) {
    double* v_saved;
    double* state_saved;
    size_t c;
    if (!sim->initialized) {
        return RCSD_OK; /* rcsd_init pins */
    }
    if (sim->geometry_dirty) {
        rcsd_build_geometry(sim);
    }
    v_saved = (double*) malloc((size_t) sim->n_nodes * sizeof(double));
    state_saved = (double*) malloc((size_t) sim->n_nodes * RCSD_NSTATE * sizeof(double));
    if (v_saved == NULL || state_saved == NULL) {
        free(v_saved);
        free(state_saved);
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    memcpy(v_saved, sim->v, (size_t) sim->n_nodes * sizeof(double));
    memcpy(state_saved, sim->state, (size_t) sim->n_nodes * RCSD_NSTATE * sizeof(double));
    for (c = 0; c < sim->cells.n; ++c) {
        init_states(sim, (int) c, sim->cells.data[c].v_hold);
    }
    eval_membrane(sim);
    for (c = 0; c < sim->cells.n; ++c) {
        pin_cell(sim, (int) c);
    }
    memcpy(sim->v, v_saved, (size_t) sim->n_nodes * sizeof(double));
    memcpy(sim->state, state_saved, (size_t) sim->n_nodes * RCSD_NSTATE * sizeof(double));
    free(v_saved);
    free(state_saved);
    return RCSD_OK;
}

/* ------------------------------------------------------------------------- */
/* recording                                                                  */
/* ------------------------------------------------------------------------- */

int rcsd_record_spikes(RCSDSim* sim, int cell) {
    if (cell < 0 || (size_t) cell >= sim->cells.n) {
        rcsd_set_error("no cell %d", cell);
        return RCSD_ERROR;
    }
    if (sim->record_spikes == NULL) {
        sim->record_spikes = (int*) calloc(sim->cells.n, sizeof(int));
        if (sim->record_spikes == NULL) {
            rcsd_set_error("out of memory");
            return RCSD_ERROR;
        }
    }
    sim->record_spikes[cell] = 1;
    return RCSD_OK;
}

static int add_trace(RCSDSim* sim, int which, int cell, int section, double dt) {
    Trace trace;
    int sec = rcsd_cell_section(sim, cell, section);
    if (sec < 0) {
        rcsd_set_error("cell %d has no section %d", cell, section);
        return RCSD_ERROR;
    }
    if (!(dt > 0.0)) {
        rcsd_set_error("recording dt must be positive");
        return RCSD_ERROR;
    }
    memset(&trace, 0, sizeof trace);
    trace.node = rcsd_section_node(sim, sec, 0.5);
    trace.dt = dt;
    /* the schedule is absolute, so a trace added mid-run starts at the
     * next grid point rather than replaying from zero */
    trace.next = (long) floor(((double) sim->step * sim->dt) / dt + 1e-9);
    if (trace.next * dt < (double) sim->step * sim->dt - 1e-9) {
        trace.next += 1;
    }
    if (which == 0) {
        DYN_PUSH(sim->v_traces, trace);
        return (int) sim->v_traces.n - 1;
    }
    DYN_PUSH(sim->i_traces, trace);
    return (int) sim->i_traces.n - 1;
}

int rcsd_record_voltage(RCSDSim* sim, int cell, int section, double dt) {
    return add_trace(sim, 0, cell, section, dt);
}

int rcsd_record_current(RCSDSim* sim, int cell, int section, double dt) {
    return add_trace(sim, 1, cell, section, dt);
}

int rcsd_clear_recordings(RCSDSim* sim) {
    size_t i;
    sim->spike_cells.n = 0;
    sim->spike_times.n = 0;
    for (i = 0; i < sim->v_traces.n; ++i) {
        sim->v_traces.data[i].values.n = 0;
    }
    for (i = 0; i < sim->i_traces.n; ++i) {
        sim->i_traces.data[i].values.n = 0;
    }
    return RCSD_OK;
}

int rcsd_spike_count(RCSDSim* sim) {
    return (int) sim->spike_times.n;
}

const int* rcsd_spike_cells(RCSDSim* sim) {
    return sim->spike_cells.data;
}

const double* rcsd_spike_times(RCSDSim* sim) {
    return sim->spike_times.data;
}

int rcsd_voltage_record_count(RCSDSim* sim) {
    return (int) sim->v_traces.n;
}

int rcsd_voltage_record_length(RCSDSim* sim, int index) {
    if (index < 0 || (size_t) index >= sim->v_traces.n) {
        return RCSD_ERROR;
    }
    return (int) sim->v_traces.data[index].values.n;
}

const double* rcsd_voltage_record(RCSDSim* sim, int index) {
    if (index < 0 || (size_t) index >= sim->v_traces.n) {
        return NULL;
    }
    return sim->v_traces.data[index].values.data;
}

int rcsd_current_record_count(RCSDSim* sim) {
    return (int) sim->i_traces.n;
}

int rcsd_current_record_length(RCSDSim* sim, int index) {
    if (index < 0 || (size_t) index >= sim->i_traces.n) {
        return RCSD_ERROR;
    }
    return (int) sim->i_traces.data[index].values.n;
}

const double* rcsd_current_record(RCSDSim* sim, int index) {
    if (index < 0 || (size_t) index >= sim->i_traces.n) {
        return NULL;
    }
    return sim->i_traces.data[index].values.data;
}

/* samples whose grid time has been reached by the simulation time */
static int record_pending(RCSDSim* sim) {
    double t = (double) sim->step * sim->dt;
    size_t i;
    for (i = 0; i < sim->v_traces.n; ++i) {
        Trace* tr = &sim->v_traces.data[i];
        while ((double) tr->next * tr->dt <= t + 1e-9) {
            DYN_PUSH(tr->values, sim->v[tr->node]);
            tr->next += 1;
        }
    }
    for (i = 0; i < sim->i_traces.n; ++i) {
        Trace* tr = &sim->i_traces.data[i];
        while ((double) tr->next * tr->dt <= t + 1e-9) {
            DYN_PUSH(tr->values, ST(tr->node, RCSD_S_IMEM));
            tr->next += 1;
        }
    }
    return RCSD_OK;
}

/* ------------------------------------------------------------------------- */
/* the timestep                                                               */
/* ------------------------------------------------------------------------- */

static void solve(RCSDSim* sim) {
    int i;
    for (i = sim->n_nodes - 1; i >= 0; --i) {
        int p = sim->parent[i];
        double q;
        if (p < 0) {
            continue;
        }
        q = sim->coef_a[i] / sim->d[i];
        sim->d[p] -= q * sim->coef_b[i];
        sim->rhs[p] -= q * sim->rhs[i];
    }
    for (i = 0; i < sim->n_nodes; ++i) {
        int p = sim->parent[i];
        if (p < 0) {
            sim->rhs[i] /= sim->d[i];
        } else {
            sim->rhs[i] -= sim->coef_b[i] * sim->rhs[p];
            sim->rhs[i] /= sim->d[i];
        }
    }
}

static int detect_spikes(RCSDSim* sim, double t_new) {
    size_t c;
    for (c = 0; c < sim->cells.n; ++c) {
        Cell* cell = &sim->cells.data[c];
        double v = sim->v[cell->soma_node];
        if (v >= cell->v_threshold) {
            double te;
            int k;
            if (cell->above) {
                continue;
            }
            cell->above = 1;
            te = t_new + 1e-10;
            /* SpikeFilter: an absolute refractory window */
            if (te - cell->t_last_spike < cell->tref) {
                continue;
            }
            cell->t_last_spike = te;
            if (sim->record_spikes && sim->record_spikes[c]) {
                DYN_PUSH(sim->spike_cells, (int) c);
                DYN_PUSH(sim->spike_times, te);
            }
            for (k = sim->out_start[c]; k < sim->out_start[c + 1]; ++k) {
                int conn = sim->out_conn[k];
                if (rcsd_events_enqueue(sim, conn, te + sim->connections.data[conn].eff_delay) !=
                    RCSD_OK) {
                    return RCSD_ERROR;
                }
            }
        } else {
            cell->above = 0;
        }
    }
    return RCSD_OK;
}

static int play_inputs(RCSDSim* sim, double horizon) {
    size_t n;
    for (n = 0; n < sim->inputs.n; ++n) {
        Input* in = &sim->inputs.data[n];
        while (in->cursor < in->n && in->times[in->cursor] <= horizon) {
            double ts = in->times[in->cursor++];
            int k;
            for (k = sim->in_start[n]; k < sim->in_start[n + 1]; ++k) {
                int conn = sim->in_conn[k];
                if (rcsd_events_enqueue(sim, conn, ts + sim->connections.data[conn].eff_delay) !=
                    RCSD_OK) {
                    return RCSD_ERROR;
                }
            }
        }
    }
    return RCSD_OK;
}

static int advance(RCSDSim* sim) {
    const double dt = sim->dt;
    const long s = sim->step;
    const double cfac = 1e-3 * sim->cj;
    double t_mid, t_new;
    int i;
#ifdef RCSD_PROFILE
    double prof_t = prof_now();
#endif

    /* 1. events queued for this step, and the external spike trains */
    if (play_inputs(sim, sim->t + 0.5 * dt) != RCSD_OK) {
        return RCSD_ERROR;
    }
    if (rcsd_events_deliver(sim, s) != RCSD_OK) {
        return RCSD_ERROR;
    }

    PROF_MARK(0);
    /* 2. BEFORE BREAKPOINT at the midpoint: the Ornstein-Uhlenbeck conductances */
    sim->t += 0.5 * dt;
    t_mid = sim->t;
    rcsd_noise_advance(sim, t_mid);

    PROF_MARK(1);
    /* 3. membrane currents and the matrix */
    eval_membrane(sim);
    PROF_MARK(2);
    rcsd_synapse_currents(sim);
    PROF_MARK(3);
    rcsd_noise_currents(sim);
    rcsd_opsin_currents(sim);
    for (i = 0; i < sim->n_nodes; ++i) {
        sim->sav_rhs[i] = -sim->rhs[i];
    }
    if (rcsd_stimulus_apply(sim, s, NULL, NULL) != RCSD_OK) {
        return RCSD_ERROR;
    }
    PROF_MARK(4);
    for (i = 0; i < sim->n_nodes; ++i) {
        if (sim->is_centre[i]) {
            sim->d[i] += cfac * PR(i, RCSD_P_CM);
        }
        sim->sav_d[i] = sim->d[i];
    }
    for (i = 0; i < sim->n_nodes; ++i) {
        int p = sim->parent[i];
        double dv;
        if (p < 0) {
            continue;
        }
        dv = sim->v[p] - sim->v[i];
        sim->rhs[i] -= sim->coef_b[i] * dv;
        sim->rhs[p] += sim->coef_a[i] * dv;
        sim->d[i] -= sim->coef_b[i];
        sim->d[p] -= sim->coef_a[i];
    }

    /* 4. solve for the half-step change, correct the ion currents to the
     *    midpoint, update the voltage and the fast membrane current */
    solve(sim);
    PROF_MARK(5);
    for (i = 0; i < sim->n_nodes; ++i) {
        double dvh = sim->rhs[i];
        if (sim->is_centre[i]) {
            ST(i, RCSD_S_INA) += sim->dinadv[i] * dvh;
            ST(i, RCSD_S_IK) += sim->dikdv[i] * dvh;
            ST(i, RCSD_S_ICA) += sim->dicadv[i] * dvh;
        }
        sim->v[i] += 2.0 * dvh;
        ST(i, RCSD_S_IMEM) = (sim->sav_rhs[i] + sim->sav_d[i] * dvh) * sim->area[i] * 0.01;
    }

    /* 5. what the per-step callback does: the field for the next step, the
     *    photon flux for this step's kinetics */
    sim->t += 0.5 * dt;
    t_new = sim->t;
    rcsd_stimulus_after_update(sim, s);

    PROF_MARK(6);
    /* 6. states at the new voltage */
    membrane_states(sim);
    PROF_MARK(7);
    rcsd_synapse_state_step(sim);
    rcsd_opsin_state_step(sim);
    PROF_MARK(8);

    /* 7. spikes */
    if (detect_spikes(sim, t_new) != RCSD_OK) {
        return RCSD_ERROR;
    }
    PROF_MARK(9);
    sim->step += 1;
    return RCSD_OK;
}

int rcsd_run(RCSDSim* sim, long n_steps, long* done) {
    long k;
    if (done) {
        *done = 0;
    }
    if (!sim->initialized) {
        rcsd_set_error("call rcsd_init before rcsd_run");
        return RCSD_ERROR;
    }
    if (sim->geometry_dirty) {
        rcsd_build_geometry(sim);
    }
    if (record_pending(sim) != RCSD_OK) {
        return RCSD_ERROR;
    }
    for (k = 0; k < n_steps; ++k) {
        int need = 0, mode = 0;
        if (rcsd_stimulus_apply(sim, sim->step, &need, &mode) != RCSD_OK) {
            return RCSD_ERROR;
        }
        if (need) {
            if (done) {
                *done = k;
            }
            return RCSD_NEED_STIMULUS;
        }
        if (k > 0 && record_pending(sim) != RCSD_OK) {
            return RCSD_ERROR;
        }
        if (advance(sim) != RCSD_OK) {
            return RCSD_ERROR;
        }
    }
    if (done) {
        *done = n_steps;
    }
    return RCSD_OK;
}
