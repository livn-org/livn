/* Saving and restoring the dynamic state of a simulation, to fork a run.
 *
 * A saved state holds everything a step reads that an earlier step wrote:
 * voltages, gating and concentration states, the per-node terms carried over
 * by the staggered scheme, synapse states and weights, the noise and opsin
 * states with their Random123 positions, the queued events, the spike
 * detector flags, the external spike-train cursors and the recording
 * schedule. Parameters set after the save are left alone, except the
 * per-site synapse table and the connections, which are restored whole as
 * NEURON restores a NET_RECEIVE mechanism. The structure (cells, sites,
 * connections, noise and opsin sites, traces) must be the one it was saved
 * from. */
#include <stdlib.h>
#include <string.h>

#include "internal.h"

struct RCSDState {
    long step;
    double t, dt, cj;
    int n_nodes;
    size_t n_cells, n_sites, sp_cap, n_conn, w_cap, n_inputs, n_noise, n_opsins, n_chr2;
    size_t n_v_traces, n_i_traces;
    int n_slots;
    double max_delay;
    /* per node */
    double* node; /* RCSD_STATE_NODE_ARRAYS arrays of n_nodes, then state */
    double* state;
    /* per cell */
    int* above;
    double* t_last_spike;
    /* per site and connection */
    double* sp;
    double* ss;
    double* sc;
    double* site_cur;
    double* site_dcur;
    double* w;
    Connection* connections;
    /* inputs, noise, opsins */
    int* cursor;
    Noise* noise;
    Opsin* opsins;
    ChR2* chr2;
    /* events: a copy of each bucket */
    EventList* buckets;
    /* recording schedule */
    long* v_next;
    long* i_next;
    long proj_next;
};

/* the per-node arrays a step carries into the next */
#define RCSD_STATE_NODE_ARRAYS 12

static double** node_arrays(RCSDSim* sim, double** out) {
    out[0] = sim->v;
    out[1] = sim->d;
    out[2] = sim->rhs;
    out[3] = sim->sav_rhs;
    out[4] = sim->sav_d;
    out[5] = sim->dinadv;
    out[6] = sim->dikdv;
    out[7] = sim->dicadv;
    out[8] = sim->ext_amp;
    out[9] = sim->stim_amp;
    out[10] = sim->stim_dens;
    out[11] = sim->stim_rhs;
    return out;
}

static void* copy_of(const void* src, size_t bytes, int* failed) {
    void* dst;
    if (bytes == 0 || src == NULL) {
        return NULL;
    }
    dst = malloc(bytes);
    if (dst == NULL) {
        *failed = 1;
        return NULL;
    }
    memcpy(dst, src, bytes);
    return dst;
}

static void copy_into(void* dst, const void* src, size_t bytes) {
    if (bytes != 0 && dst != NULL && src != NULL) {
        memcpy(dst, src, bytes);
    }
}

void rcsd_state_free(RCSDState* s) {
    size_t i;
    if (s == NULL) {
        return;
    }
    free(s->node);
    free(s->state);
    free(s->above);
    free(s->t_last_spike);
    free(s->sp);
    free(s->ss);
    free(s->sc);
    free(s->site_cur);
    free(s->site_dcur);
    free(s->w);
    free(s->connections);
    free(s->cursor);
    free(s->noise);
    free(s->opsins);
    free(s->chr2);
    if (s->buckets != NULL) {
        for (i = 0; i < (size_t) s->n_slots; ++i) {
            free(s->buckets[i].data);
        }
        free(s->buckets);
    }
    free(s->v_next);
    free(s->i_next);
    free(s);
}

RCSDState* rcsd_state_save(RCSDSim* sim) {
#if defined(RCSD_OPENACC)
    (void) sim;
    rcsd_set_error("the OpenACC build cannot save a simulation state");
    return NULL;
#else
    RCSDState* s;
    double* arrays[RCSD_STATE_NODE_ARRAYS];
    size_t n, i;
    int k, failed = 0;
    if (!sim->initialized) {
        rcsd_set_error("nothing to save: the simulation has not been initialised");
        return NULL;
    }
    s = (RCSDState*) calloc(1, sizeof(RCSDState));
    if (s == NULL) {
        rcsd_set_error("out of memory");
        return NULL;
    }
    s->step = sim->step;
    s->t = sim->t;
    s->dt = sim->dt;
    s->cj = sim->cj;
    s->n_nodes = sim->n_nodes;
    s->n_cells = sim->cells.n;
    s->n_sites = sim->synapses.n;
    s->sp_cap = sim->sp_cap;
    s->n_conn = sim->connections.n;
    s->w_cap = sim->w_cap;
    s->n_inputs = sim->inputs.n;
    s->n_noise = sim->noise.n;
    s->n_opsins = sim->opsins.n;
    s->n_chr2 = sim->chr2.n;
    s->n_v_traces = sim->v_traces.n;
    s->n_i_traces = sim->i_traces.n;
    s->n_slots = sim->n_slots;
    s->max_delay = sim->max_delay;

    n = (size_t) sim->n_nodes;
    s->node = (double*) calloc(RCSD_STATE_NODE_ARRAYS * (n ? n : 1), sizeof(double));
    failed |= s->node == NULL;
    node_arrays(sim, arrays);
    for (k = 0; k < RCSD_STATE_NODE_ARRAYS && !failed; ++k) {
        copy_into(s->node + (size_t) k * n, arrays[k], n * sizeof(double));
    }
    s->state = (double*) copy_of(sim->state, n * RCSD_NSTATE * sizeof(double), &failed);

    if (s->n_cells) {
        s->above = (int*) malloc(s->n_cells * sizeof(int));
        s->t_last_spike = (double*) malloc(s->n_cells * sizeof(double));
        failed |= s->above == NULL || s->t_last_spike == NULL;
        for (i = 0; i < s->n_cells && !failed; ++i) {
            s->above[i] = sim->cells.data[i].above;
            s->t_last_spike[i] = sim->cells.data[i].t_last_spike;
        }
    }

    s->sp = (double*) copy_of(sim->sp, s->sp_cap * RCSD_SP_N * sizeof(double), &failed);
    s->ss = (double*) copy_of(sim->ss, s->sp_cap * RCSD_SS_N * sizeof(double), &failed);
    s->sc = (double*) copy_of(sim->sc, s->sp_cap * SYN_CACHE_N * sizeof(double), &failed);
    s->site_cur = (double*) copy_of(sim->site_cur, s->sp_cap * sizeof(double), &failed);
    s->site_dcur = (double*) copy_of(sim->site_dcur, s->sp_cap * sizeof(double), &failed);
    s->w = (double*) copy_of(sim->w, s->n_conn * RCSD_NWEIGHT * sizeof(double), &failed);
    s->connections = (Connection*) copy_of(sim->connections.data,
                                           s->n_conn * sizeof(Connection), &failed);

    if (s->n_inputs) {
        s->cursor = (int*) malloc(s->n_inputs * sizeof(int));
        failed |= s->cursor == NULL;
        for (i = 0; i < s->n_inputs && !failed; ++i) {
            s->cursor[i] = sim->inputs.data[i].cursor;
        }
    }
    s->noise = (Noise*) copy_of(sim->noise.data, s->n_noise * sizeof(Noise), &failed);
    s->opsins = (Opsin*) copy_of(sim->opsins.data, s->n_opsins * sizeof(Opsin), &failed);
    s->chr2 = (ChR2*) copy_of(sim->chr2.data, s->n_chr2 * sizeof(ChR2), &failed);

    if (sim->buckets != NULL && s->n_slots > 0) {
        s->buckets = (EventList*) calloc((size_t) s->n_slots, sizeof(EventList));
        failed |= s->buckets == NULL;
        for (k = 0; k < s->n_slots && !failed; ++k) {
            const EventList* b = &sim->buckets[k];
            s->buckets[k].n = s->buckets[k].cap = b->n;
            s->buckets[k].data = (Event*) copy_of(b->data, b->n * sizeof(Event), &failed);
        }
    }

    if (s->n_v_traces) {
        s->v_next = (long*) malloc(s->n_v_traces * sizeof(long));
        failed |= s->v_next == NULL;
        for (i = 0; i < s->n_v_traces && !failed; ++i) {
            s->v_next[i] = sim->v_traces.data[i].next;
        }
    }
    if (s->n_i_traces) {
        s->i_next = (long*) malloc(s->n_i_traces * sizeof(long));
        failed |= s->i_next == NULL;
        for (i = 0; i < s->n_i_traces && !failed; ++i) {
            s->i_next[i] = sim->i_traces.data[i].next;
        }
    }
    s->proj_next = sim->proj_next;

    if (failed) {
        rcsd_state_free(s);
        rcsd_set_error("out of memory");
        return NULL;
    }
    return s;
#endif
}

static int mismatch(const char* what, size_t saved, size_t now) {
    rcsd_set_error("the state was saved with %zu %s and the simulation now has %zu; a state "
                   "restores only into the network it was saved from",
                   saved, what, now);
    return RCSD_ERROR;
}

int rcsd_state_restore(RCSDSim* sim, const RCSDState* s) {
#if defined(RCSD_OPENACC)
    (void) sim;
    (void) s;
    rcsd_set_error("the OpenACC build cannot restore a simulation state");
    return RCSD_ERROR;
#else
    double* arrays[RCSD_STATE_NODE_ARRAYS];
    size_t n, i;
    int k;
    if (s == NULL) {
        rcsd_set_error("no state to restore");
        return RCSD_ERROR;
    }
    if ((size_t) sim->n_nodes != (size_t) s->n_nodes) {
        return mismatch("nodes", (size_t) s->n_nodes, (size_t) sim->n_nodes);
    }
    if (sim->cells.n != s->n_cells) return mismatch("cells", s->n_cells, sim->cells.n);
    if (sim->synapses.n != s->n_sites) return mismatch("sites", s->n_sites, sim->synapses.n);
    if (sim->sp_cap != s->sp_cap) return mismatch("site slots", s->sp_cap, sim->sp_cap);
    if (sim->connections.n != s->n_conn) {
        return mismatch("connections", s->n_conn, sim->connections.n);
    }
    if (sim->inputs.n != s->n_inputs) return mismatch("inputs", s->n_inputs, sim->inputs.n);
    if (sim->noise.n != s->n_noise) return mismatch("noise sites", s->n_noise, sim->noise.n);
    if (sim->opsins.n != s->n_opsins) return mismatch("opsins", s->n_opsins, sim->opsins.n);
    if (sim->chr2.n != s->n_chr2) return mismatch("ChR2 sites", s->n_chr2, sim->chr2.n);
    if (sim->v_traces.n != s->n_v_traces) {
        return mismatch("voltage traces", s->n_v_traces, sim->v_traces.n);
    }
    if (sim->i_traces.n != s->n_i_traces) {
        return mismatch("current traces", s->n_i_traces, sim->i_traces.n);
    }
    if (sim->geometry_dirty) {
        rcsd_set_error("the geometry changed since the state was saved");
        return RCSD_ERROR;
    }
    if (!sim->wired && rcsd_wire(sim) != RCSD_OK) {
        return RCSD_ERROR;
    }

    /* the event ring first: the only step that can fail for want of memory */
    if (s->n_slots != sim->n_slots) {
        EventList* ring = (EventList*) calloc((size_t) (s->n_slots > 0 ? s->n_slots : 1),
                                              sizeof(EventList));
        if (ring == NULL) {
            rcsd_set_error("out of memory");
            return RCSD_ERROR;
        }
        if (sim->buckets != NULL) {
            for (k = 0; k < sim->n_slots; ++k) {
                free(sim->buckets[k].data);
            }
            free(sim->buckets);
        }
        sim->buckets = ring;
        sim->n_slots = s->n_slots;
    }
    for (k = 0; k < s->n_slots; ++k) {
        EventList* b = &sim->buckets[k];
        const EventList* saved = &s->buckets[k];
        if (b->cap < saved->n &&
            !dyn_reserve((void**) &b->data, &b->cap, saved->n, sizeof(Event))) {
            return RCSD_ERROR;
        }
        copy_into(b->data, saved->data, saved->n * sizeof(Event));
        b->n = saved->n;
    }
    sim->max_delay = s->max_delay;

    sim->step = s->step;
    sim->t = s->t;
    sim->dt = s->dt;
    sim->cj = s->cj;
    sim->initialized = 1;

    n = (size_t) sim->n_nodes;
    node_arrays(sim, arrays);
    for (k = 0; k < RCSD_STATE_NODE_ARRAYS; ++k) {
        copy_into(arrays[k], s->node + (size_t) k * n, n * sizeof(double));
    }
    copy_into(sim->state, s->state, n * RCSD_NSTATE * sizeof(double));

    for (i = 0; i < s->n_cells; ++i) {
        sim->cells.data[i].above = s->above[i];
        sim->cells.data[i].t_last_spike = s->t_last_spike[i];
    }
    copy_into(sim->sp, s->sp, s->sp_cap * RCSD_SP_N * sizeof(double));
    copy_into(sim->ss, s->ss, s->sp_cap * RCSD_SS_N * sizeof(double));
    copy_into(sim->sc, s->sc, s->sp_cap * SYN_CACHE_N * sizeof(double));
    copy_into(sim->site_cur, s->site_cur, s->sp_cap * sizeof(double));
    copy_into(sim->site_dcur, s->site_dcur, s->sp_cap * sizeof(double));
    copy_into(sim->w, s->w, s->n_conn * RCSD_NWEIGHT * sizeof(double));
    copy_into(sim->connections.data, s->connections, s->n_conn * sizeof(Connection));
    for (i = 0; i < s->n_inputs; ++i) {
        sim->inputs.data[i].cursor = s->cursor[i];
    }
    copy_into(sim->noise.data, s->noise, s->n_noise * sizeof(Noise));
    copy_into(sim->opsins.data, s->opsins, s->n_opsins * sizeof(Opsin));
    copy_into(sim->chr2.data, s->chr2, s->n_chr2 * sizeof(ChR2));

    for (i = 0; i < s->n_v_traces; ++i) {
        sim->v_traces.data[i].next = s->v_next[i];
        sim->v_traces.data[i].values.n = 0;
    }
    for (i = 0; i < s->n_i_traces; ++i) {
        sim->i_traces.data[i].next = s->i_next[i];
        sim->i_traces.data[i].values.n = 0;
    }
    sim->proj_next = s->proj_next;
    sim->proj_values.n = 0;
    sim->spike_cells.n = 0;
    sim->spike_times.n = 0;
    return RCSD_OK;
#endif
}

double rcsd_state_time(const RCSDState* s) {
    return s == NULL ? 0.0 : s->t;
}
