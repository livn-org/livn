/* Internal data layout of librcsd. Not part of the public API. */
#ifndef RCSD_INTERNAL_H
#define RCSD_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "rcsd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* --- OpenACC offload ------------------------------------------------------------ */
/* RCSD_OPENACC builds the per-node, per-site and per-cell loops as GPU
 * kernels. It assumes memory the device can address without data clauses:
 * coherent unified memory on a Grace Hopper (nvc -gpu=mem:unified), or
 * managed memory on a discrete GPU (-gpu=mem:managed), where every heap
 * allocation the library makes is managed. So a kernel names the pointers it
 * dereferences as deviceptr and nothing is copied; what stays on the host
 * (stimulus rows, recording, external spike trains) touches the same memory.
 *
 * Inside rcsd_run every kernel goes on one async queue (sim->acc_async), and
 * the host waits on it only where it reads or writes what the kernels use:
 * the spike history it collects the recorded spikes from, a voltage trace
 * that is due, an active stimulus, an external spike train's events. Spikes
 * never come back to the host to make events: the device keeps the cells
 * that fired at each recent step and pulls each step's events out of that
 * history (see rcsd_events_deliver). Outside rcsd_run, acc_async is
 * acc_async_sync and every kernel is synchronous.
 *
 * The device rounds exp and pow differently from the host libm and may fuse
 * multiply-adds, so a GPU build does not reproduce NEURON to the bit the way
 * the CPU build does; nor does it sum a site's events in the CPU build's
 * order. One thread per iteration, and every loop inside a kernel
 * sequential (nvc -acc=noautopar): a node's currents are summed in NEURON's
 * order on the device too, not as a parallel reduction. Without RCSD_OPENACC
 * the macros expand to nothing. */
#if defined(RCSD_OPENACC)
#include <openacc.h>
#define RCSD_PRAGMA_(x) _Pragma(#x)
#define RCSD_ACC_LOOP(...) \
    RCSD_PRAGMA_(acc parallel loop gang vector deviceptr(__VA_ARGS__) async(sim->acc_async))
#define RCSD_ACC_GANG(...) \
    RCSD_PRAGMA_(acc parallel loop gang deviceptr(__VA_ARGS__) async(sim->acc_async))
#define RCSD_ACC_SERIAL(...) RCSD_PRAGMA_(acc serial deviceptr(__VA_ARGS__) async(sim->acc_async))
#define RCSD_ACC_VECTOR RCSD_PRAGMA_(acc loop vector)
#define RCSD_ACC_VECTOR_SUM(x) RCSD_PRAGMA_(acc loop vector reduction(+ : x))
#define RCSD_ACC_ROUTINE RCSD_PRAGMA_(acc routine seq)
#define RCSD_ACC_SEQ RCSD_PRAGMA_(acc loop seq)
#define RCSD_ACC_ATOMIC_CAPTURE RCSD_PRAGMA_(acc atomic capture)
#define RCSD_ACC_ATOMIC_UPDATE RCSD_PRAGMA_(acc atomic update)
#define RCSD_ACC_WAIT RCSD_PRAGMA_(acc wait)
/* node state and parameters a variable at a time, cap_nodes apart, so the
 * threads of a warp, a node each, read adjacent memory */
#define RCSD_STATE(sim, node, s) \
    (sim)->state[(size_t) (s) * (size_t) (sim)->cap_nodes + (size_t) (node)]
#define RCSD_PARAM(sim, node, p) \
    (sim)->param[(size_t) (p) * (size_t) (sim)->cap_nodes + (size_t) (node)]
/* how often rcsd_run collects the recorded spikes from the spike history */
#define RCSD_HARVEST_STEPS 64
#else
/* a node's state and parameters together, which is what a CPU thread
 * walking the nodes in order wants */
#define RCSD_STATE(sim, node, s) (sim)->state[(size_t) (node) * RCSD_NSTATE + (size_t) (s)]
#define RCSD_PARAM(sim, node, p) (sim)->param[(size_t) (node) * RCSD_NPARAM + (size_t) (p)]
#define RCSD_ACC_LOOP(...)
#define RCSD_ACC_GANG(...)
#define RCSD_ACC_SERIAL(...)
#define RCSD_ACC_VECTOR
#define RCSD_ACC_VECTOR_SUM(x)
#define RCSD_ACC_ROUTINE
#define RCSD_ACC_SEQ
#define RCSD_ACC_ATOMIC_CAPTURE
#define RCSD_ACC_ATOMIC_UPDATE
#define RCSD_ACC_WAIT
#endif

/* --- physical constants, as NEURON compiles them (CODATA 2018) --------------- */
/* h.FARADAY and h.R of NEURON 9 (CODATA 2018), which nrn_nernst and the
 * FARADAY of the .mod files share to the last digit */
#define RCSD_FARADAY 96485.33212331001
#define RCSD_GASCONSTANT 8.31446261815324

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
/* NEURON's Pt3d: single-precision coordinates and diameter, double arc */
typedef struct {
    float x, y, z, d;
    double arc;
} Pt3d;

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
    Pt3d* pt3d; /* the 3-D points define_shape gave the section, if any */
    int npt3d, pt3d_cap;
    int recalc_area;
} Section;

typedef struct {
    int gid;
    int population;
    int sec0, nsec;
    int node0, nnode;
    int root_node;
    int soma_section;
    int soma_node; /* where spikes are detected: the soma unless a detector is set */
    double v_threshold, v_hold, tref;
    double t_last_spike; /* SpikeFilter.t_last */
    int above;           /* PreSyn.flag_ */
    int opsin;
} Cell;

/* --- point processes, in NEURON's mechanism-type order ------------------------- */
/* nrn_rhs and nrn_lhs visit the mechanisms of a node in the order of their
 * type index, density and point mechanisms interleaved, and the instances of
 * one type at a node in reverse order of creation (prop_alloc links at the
 * head). The sum of the currents and conductances of a node is taken in that
 * order so that it rounds as NEURON's does; each slot is one point-process
 * type at its position among the density mechanisms of mech.h. */
enum {
    PP_ICLAMP = 0, /* IClamp (7): the current stimuli */
    PP_GFLUCT,     /* Gfluct3 (32) */
    PP_LINEXP2,    /* LinExp2Syn (40) */
    PP_NMDA,       /* LinExp2SynNMDA (41) */
    PP_RHO3C,      /* RhO3c (46) */
    PP_CHR2,       /* ChR2H134R (30): a density mechanism, summed after CaN */
    PP_STDP_INH,   /* StdpLinExp2SynInh (48) */
    PP_STDP,       /* StdpLinExp2Syn (49) */
    PP_STDP_NMDA,  /* StdpLinExp2SynNMDA (50) */
    PP_N
};

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
    int site; /* the connection's, so the OpenACC build can group a bucket by it */
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
    double cur, dcur; /* this step's current and dI/dV, scaled by 1e2/area */
    R123Stream stream;
    uint32_t id1, id2, id3;
    int seeded;
} Noise;

/* --- opsin (RhO3c) ------------------------------------------------------------------ */
typedef struct {
    int cell, section, node;
    double g0, E, v0, v1, k_a, k_r, p, q, Gd, Gr0, phi_m;
    double C, O, phi;
    double cur, dcur; /* this step's current and dI/dV, scaled by 1e2/area */
} Opsin;

/* --- opsin (ChR2H134R) -------------------------------------------------------------- */
typedef struct {
    int cell, section, node;
    double param[RCSD_CHR2_NPARAM];
    double O1, O2, C1, p;
    double irr;       /* Irradiance, mW/mm2 */
    double cur, dcur; /* this step's current density (mA/cm2) and dI/dV */
} ChR2;

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

/* a row of samples a recording group wrote at stage_val + at */
typedef struct {
    int group;
    size_t at;
} StageRow;

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
    double* state;   /* RCSD_NSTATE per node, v excluded (kept in v[]); see RCSD_STATE */
    double* param;   /* RCSD_NPARAM per node; see RCSD_PARAM */
    unsigned* mech;
    double* dinadv;
    double* dikdv;
    double* dicadv;
    double* ext_amp;   /* nA, extracellular equivalent current, set for the next step */
    double* stim_amp;  /* nA, current-mode injection for this step */
    double* stim_dens; /* mA/cm2, current-density injection for this step */
    double* stim_rhs;  /* what the current stimuli add to rhs this step (the IClamp slot) */
    double* hold_amp;  /* nA, a steady holding current per node: an IClamp held on,
                        * created before any stimulus's, so summed ahead of them */
    int tile_nodes;    /* n_nodes when cells_tile_nodes last looked, -1 to look again */
    int tiled;         /* what it found: one tree per cell, cells end to end */
    /* the OpenACC build. The kernels' queue inside rcsd_run. The spike
     * history: for the last hist_len steps, which cells fired (hist_cells
     * slots a step), when, and which step a slot holds (-1: none); the
     * recorded spikes are collected from it up to step `harvested`. Each
     * cell's outgoing connections sorted by out_off, their delay in steps,
     * which lies in [min_off, max_off]. Where each site's events start in a
     * step's sorted bucket of external spike-train events. */
    int acc_async;
    int hist_len, hist_cells;
    int* hist;
    int* hist_n;
    double* hist_te;
    long* hist_step;
    long harvested;
    int* out_off;
    int min_off, max_off;
    int* seg;
    size_t seg_cap;
    /* recording on the device. The traces, grouped at the start of a run by
     * the dt and next sample they share, so a group comes due as one: rec_node
     * holds each member's node (-(node + 1) for a membrane current) and
     * rec_trace its trace (a v_traces index, or -(i_traces index + 1)), the
     * members of group g from rec_start[g]. A due group is a row of samples a
     * kernel writes into stage_val at stage_n; the rows rcsd_run has not yet
     * collected are listed in stage_rows */
    int* rec_node;
    int* rec_trace;
    int* rec_start;
    long* rec_next;
    double* rec_dt;
    int n_rec, n_rec_groups;
    double* stage_val;
    size_t stage_n, stage_cap;
    DYN(StageRow) stage_rows;
    /* each opsin's and ChR2 segment's photon-flux row (rcsd_stimulus_prepare) */
    int* opsin_row;
    int* chr2_row;

    /* point-process instances per node and slot, in NEURON's order */
    int* pp_start[PP_N]; /* [n_nodes + 1] */
    int* pp_list[PP_N];
    int pp_nodes;
    int pp_dirty;

    /* synapses */
    DYN(Synapse) synapses;
    double* sp; /* [n_sites][RCSD_SP_N] */
    double* ss; /* [n_sites][RCSD_SS_N] */
    double* sc; /* [n_sites][SYN_CACHE_N]: the per-step decay factors, which depend only on dt and the taus */
    double* site_cur;  /* this step's current per site, scaled by 1e2/area */
    double* site_dcur; /* and its dI/dV */
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
    /* per section, 1 + its index in `noise`, 0 where it carries none */
    int* noise_at;
    size_t noise_at_n;
    DYN(Opsin) opsins;
    DYN(ChR2) chr2;

    /* stimulus */
    Stimulus stim[RCSD_STIM_N];

    /* recording */
    DYN(int) spike_cells;
    DYN(double) spike_times;
    int* record_spikes; /* per cell */
    DYN(Trace) v_traces;
    DYN(Trace) i_traces;
    /* the projected recording: channel c is the weighted sum of the membrane
     * currents at proj_node[proj_start[c] .. proj_start[c + 1]), sampled
     * every proj_dt into proj_values, one row of n_proj values a sample */
    int n_proj;
    int* proj_start;
    int* proj_node;
    double* proj_weight;
    double proj_dt;
    long proj_next;
    DYN(double) proj_values;
    /* arrival buckets: a spike's connections, filed when the spike is new
     * under the step each event arrives at, so a step reads only its own.
     * arr_len buckets of arr_cap; one counted past arr_cap overflowed and
     * that step scans the history instead (arr_overflow counts them, and
     * the next harvest grows the cap). arr_next: the step the buckets are
     * filed for, -1 to refile them from the history */
    int arr_len, arr_cap;
    int* arr_n;
    int* arr_conn;
    double* arr_te;
    int* arr_overflow;
    long arr_next;
};

/* shared helpers */
void rcsd_set_error(const char* fmt, ...);
int rcsd_alloc_nodes(RCSDSim* sim, int n);
int rcsd_build_geometry(RCSDSim* sim);
int rcsd_pp_index(RCSDSim* sim);
int rcsd_wire(RCSDSim* sim);
void rcsd_synapse_init_states(RCSDSim* sim);
/* shape.c: NEURON's 3-D points and the areas and resistances they imply */
int rcsd_shape_define(RCSDSim* sim);
void rcsd_shape_free(Section* sec);
void rcsd_shape_length_change(Section* sec, double L);
void rcsd_shape_diam_change(Section* sec, double diam);
void rcsd_shape_area_ri(RCSDSim* sim, Section* sec);

void rcsd_noise_init(RCSDSim* sim);
void rcsd_noise_advance(RCSDSim* sim, double t_mid);
void rcsd_noise_currents(RCSDSim* sim);
void rcsd_opsin_init(RCSDSim* sim);
void rcsd_opsin_state_step(RCSDSim* sim);
void rcsd_opsin_currents(RCSDSim* sim);
void rcsd_chr2_init(RCSDSim* sim);
void rcsd_chr2_state_step(RCSDSim* sim);
void rcsd_chr2_currents(RCSDSim* sim);
void rcsd_stimulus_free(Stimulus* s);
int rcsd_stimulus_apply(RCSDSim* sim, long step, int* need_refill, int* need_mode);
void rcsd_stimulus_after_update(RCSDSim* sim, long step);
int rcsd_stimulus_prepare(RCSDSim* sim);
int rcsd_events_enqueue(RCSDSim* sim, int conn, double te);
void rcsd_events_clear(RCSDSim* sim);
int rcsd_events_deliver(RCSDSim* sim, long step);
int rcsd_events_harvest(RCSDSim* sim);
void rcsd_synapse_currents(RCSDSim* sim);
void rcsd_synapse_state_step(RCSDSim* sim);
void rcsd_synapse_finalize_factor(RCSDSim* sim, int site);

void r123_seed(R123Stream* s, uint32_t id1, uint32_t id2, uint32_t id3);
void r123_setseq(R123Stream* s, uint32_t seq, int which);
/* drawn on the device by the noise kernel */
RCSD_ACC_ROUTINE
double r123_normal(R123Stream* s);

#endif
