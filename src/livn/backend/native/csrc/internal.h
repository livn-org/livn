/* Internal data layout of librcsd. Not part of the public API. */
#ifndef RCSD_INTERNAL_H
#define RCSD_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "rcsd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --- physical constants, as NEURON compiles them (CODATA 2018) --------------- */
#define RCSD_FARADAY 96485.33212
#define RCSD_GASCONSTANT 8.314462618

/* --- growable arrays ----------------------------------------------------------- */
#define DYN(type)      \
    struct {           \
        type* data;    \
        size_t n, cap; \
    }

int dyn_reserve(void** data, size_t* cap, size_t n, size_t itemsize);

#define DYN_PUSH(arr, value)                                                    \
    do {                                                                        \
        if ((arr).n == (arr).cap) {                                             \
            if (!dyn_reserve((void**) &(arr).data, &(arr).cap, (arr).n + 1,     \
                             sizeof(*(arr).data)))                              \
                return RCSD_ERROR;                                              \
        }                                                                       \
        (arr).data[(arr).n++] = (value);                                        \
    } while (0)

#define DYN_FREE(arr)          \
    do {                       \
        free((arr).data);      \
        (arr).data = NULL;     \
        (arr).n = (arr).cap = 0; \
    } while (0)

/* --- cells and sections --------------------------------------------------------- */
typedef struct {
    int cell;
    int kind;
    int nseg;
    int node0;      /* first centre node; centre nodes are contiguous */
    int end_node;   /* zero-area node at the 1 end */
    int parent_section;
    double parent_x;
    int parent_node; /* node the first centre node couples to */
    double L, diam, Ra;
    unsigned mech;
} Section;

typedef struct {
    int gid;
    int population;
    int sec0, nsec;
    int node0, nnode;
    int root_node;
    int soma_section;
    int soma_node; /* where spikes are detected */
    double v_threshold, v_hold, tref;
    double t_last_spike; /* SpikeFilter.t_last */
    int above;           /* PreSyn.flag_ */
    int opsin;
} Cell;

/* --- synapses -------------------------------------------------------------------- */
enum { SC_EXP_RISE = 0, SC_EXP_DECAY, SC_HALF_RISE, SC_HALF_DECAY, SC_EXP_LEARN, SYN_CACHE_N };

typedef struct {
    int node;
    int cell;
    int kind;
} Synapse;

typedef struct {
    int source;  /* cell index, or -(input + 1) */
    int site;
    double delay;     /* physical, ms */
    double eff_delay; /* max(delay, 2 dt) once dt is known */
} Connection;

typedef struct {
    int gid;
    double* times;
    int n;
    int cursor;
    int out0, nout; /* outgoing connections */
} Input;

typedef struct {
    int conn;
    double te;
} Event;

typedef DYN(Event) EventList;

/* --- noise (Gfluct3) -------------------------------------------------------------- */
typedef struct {
    uint32_t c[4];
    uint32_t k[2];
    uint32_t r[4];
    int which;
} R123Stream;

typedef struct {
    int cell, section, node;
    double g_e0, g_i0, std_e, std_i, tau_e, tau_i, E_e, E_i, h;
    int on;
    double g_e1, g_i1, exp_e, exp_i, amp_e, amp_i, t_last;
    double g_e, g_i, ival;
    R123Stream stream;
    uint32_t id1, id2, id3;
    int seeded;
} Noise;

/* --- opsin (RhO3c) ------------------------------------------------------------------ */
typedef struct {
    int cell, section, node;
    double g0, E, v0, v1, k_a, k_r, p, q, Gd, Gr0, phi_m;
    double C, O, phi;
} Opsin;

/* --- stimulus ------------------------------------------------------------------------ */
typedef struct {
    int n_rows;
    int* cell;
    int* section;
    int* node;      /* node a row lands on (non-extracellular modes) */
    double stim_dt;
    long first;     /* first sample index held */
    int n_samples;
    double* values; /* [n_samples][n_rows] */
    long extent;    /* total samples the stimulus has; beyond it is zero */
    int active;     /* rows declared */
    /* extracellular junctions: current a = (V_b - V_a) * inv_r into node_a, out of node_b */
    int n_junctions;
    int* j_row_a;
    int* j_row_b;
    int* j_node_a;
    int* j_node_b;
    double* j_inv_r;
    int dropped_junctions;
    long needed; /* sample index a step needed and did not have, else -1 */
} Stimulus;

/* --- recording ---------------------------------------------------------------------- */
typedef struct {
    int node;
    double dt;
    long next; /* next sample index k (time k*dt) */
    DYN(double) values;
} Trace;

/* --- the simulation ------------------------------------------------------------------ */
struct RCSDSim {
    double celsius;
    double v_init;
    double dt;
    double cj; /* 2/dt for the staggered Crank-Nicolson scheme */
    long step;
    double t;  /* NEURON's clock: accumulated in half steps, so it drifts like NEURON's */
    int initialized;
    int geometry_dirty;

    DYN(Cell) cells;
    DYN(Section) sections;

    /* nodes */
    int n_nodes;
    int cap_nodes;
    int* parent;
    int* section_of; /* -1 for a zero-area node */
    int* is_centre;
    double* area;    /* um2; 100 for zero-area nodes */
    double* rinv;    /* uS, conductance to the parent node */
    double* coef_a;  /* NODEA */
    double* coef_b;  /* NODEB */
    double* d;
    double* rhs;
    double* sav_rhs;
    double* sav_d;
    double* v;
    double* state;   /* [n_nodes][RCSD_NSTATE], v excluded (kept in v[]) */
    double* param;   /* [n_nodes][RCSD_NPARAM] */
    unsigned* mech;
    double* dinadv;
    double* dikdv;
    double* dicadv;
    double* ext_amp;   /* nA, extracellular equivalent current, set for the next step */
    double* stim_amp;  /* nA, current-mode injection for this step */
    double* stim_dens; /* mA/cm2, current-density injection for this step */

    /* synapses */
    DYN(Synapse) synapses;
    double* sp; /* [n_sites][RCSD_SP_N] */
    double* ss; /* [n_sites][RCSD_SS_N] */
    double* sc; /* [n_sites][SYN_CACHE_N]: the per-step decay factors, which depend only on dt and the taus */
    size_t sp_cap;
    DYN(Connection) connections;
    double* w; /* [n_conn][RCSD_NWEIGHT] */
    size_t w_cap;
    DYN(Input) inputs;
    int* out_start; /* per cell, into out_conn */
    int* out_conn;
    int* in_start;  /* per input, into in_conn */
    int* in_conn;
    int wired;      /* out_* built for the current connection count */

    /* events */
    int n_slots;
    EventList* buckets;
    double max_delay;

    /* noise, opsins */
    DYN(Noise) noise;
    DYN(Opsin) opsins;

    /* stimulus */
    Stimulus stim[RCSD_STIM_N];

    /* recording */
    DYN(int) spike_cells;
    DYN(double) spike_times;
    int* record_spikes; /* per cell */
    DYN(Trace) v_traces;
    DYN(Trace) i_traces;
};

/* shared helpers */
void rcsd_set_error(const char* fmt, ...);
int rcsd_alloc_nodes(RCSDSim* sim, int n);
int rcsd_build_geometry(RCSDSim* sim);
int rcsd_wire(RCSDSim* sim);
void rcsd_synapse_init_states(RCSDSim* sim);
void rcsd_noise_init(RCSDSim* sim);
void rcsd_noise_advance(RCSDSim* sim, double t_mid);
void rcsd_noise_currents(RCSDSim* sim);
void rcsd_opsin_init(RCSDSim* sim);
void rcsd_opsin_state_step(RCSDSim* sim);
void rcsd_opsin_currents(RCSDSim* sim);
void rcsd_stimulus_free(Stimulus* s);
int rcsd_stimulus_apply(RCSDSim* sim, long step, int* need_refill, int* need_mode);
void rcsd_stimulus_after_update(RCSDSim* sim, long step);
int rcsd_events_enqueue(RCSDSim* sim, int conn, double te);
void rcsd_events_clear(RCSDSim* sim);
int rcsd_events_deliver(RCSDSim* sim, long step);
void rcsd_synapse_receive(RCSDSim* sim, int conn, double te);
void rcsd_synapse_currents(RCSDSim* sim);
void rcsd_synapse_state_step(RCSDSim* sim);
void rcsd_synapse_finalize_factor(RCSDSim* sim, int site);

void r123_seed(R123Stream* s, uint32_t id1, uint32_t id2, uint32_t id3);
void r123_setseq(R123Stream* s, uint32_t seq, int which);
double r123_normal(R123Stream* s);

#endif
