/* Synaptic point processes, connections, external inputs and the event queue.
 *
 * A site is one point process on a postsynaptic node, shared by every
 * connection that arrives there (LinExp2Syn's per-stream linear summation).
 * A connection carries its NetCon weight vector: `weight`, `g_unit`, and the
 * per-connection state the mechanism keeps there (`w_plastic`/`last_int` for
 * the STDP variants, `R`/`tlast` for the depressing one).
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "mech.h"

/* Structure of arrays: column p of every site is contiguous, with the
 * capacity as the stride, so the per-step passes stream only the columns
 * they read instead of every site's whole row. */
#define SP(site, p) sim->sp[(size_t) (p) * sim->sp_cap + (site)]
#define SS(site, s) sim->ss[(size_t) (s) * sim->sp_cap + (site)]
#define W(conn, k) sim->w[(size_t) (conn) * RCSD_NWEIGHT + (k)]
#define SC(site, c) sim->sc[(size_t) (c) * sim->sp_cap + (site)]

/* grow one column-major table from `old_cap` to `cap` sites, keeping the
 * `n` rows in use of each of its `columns` */
static double* grow_table(double* old, size_t columns, size_t n, size_t old_cap, size_t cap) {
    double* fresh = (double*) calloc(columns * cap, sizeof(double));
    size_t c;
    if (fresh == NULL) {
        return NULL;
    }
    if (old != NULL) {
        for (c = 0; c < columns; ++c) {
            memcpy(fresh + c * cap, old + c * old_cap, n * sizeof(double));
        }
        free(old);
    }
    return fresh;
}

static int grow_sites(RCSDSim* sim, size_t n) {
    size_t cap = sim->sp_cap ? sim->sp_cap : 64;
    size_t used = sim->synapses.n;
    double* sp;
    double* ss;
    double* sc;
    if (n <= sim->sp_cap) {
        return RCSD_OK;
    }
    while (cap < n) {
        cap *= 2;
    }
    sp = grow_table(sim->sp, RCSD_SP_N, used, sim->sp_cap, cap);
    if (sp == NULL) {
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    sim->sp = sp;
    ss = grow_table(sim->ss, RCSD_SS_N, used, sim->sp_cap, cap);
    if (ss == NULL) {
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    sim->ss = ss;
    sc = grow_table(sim->sc, SYN_CACHE_N, used, sim->sp_cap, cap);
    if (sc == NULL) {
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    sim->sc = sc;
    sim->sp_cap = cap;
    return RCSD_OK;
}

static int is_stdp(int kind) {
    return kind == RCSD_SYN_STDP || kind == RCSD_SYN_STDP_NMDA || kind == RCSD_SYN_STDP_INH;
}

static int is_nmda(int kind) {
    return kind == RCSD_SYN_NMDA || kind == RCSD_SYN_STDP_NMDA;
}

int rcsd_add_synapse(RCSDSim* sim, int cell, int section, double x, int kind) {
    Synapse syn;
    int sec, site;
    if (kind < RCSD_SYN_LINEXP2 || kind > RCSD_SYN_STDP_INH) {
        rcsd_set_error("unknown synapse kind %d", kind);
        return RCSD_ERROR;
    }
    sec = rcsd_cell_section(sim, cell, section);
    if (sec < 0) {
        rcsd_set_error("cell %d has no section %d", cell, section);
        return RCSD_ERROR;
    }
    if (grow_sites(sim, sim->synapses.n + 1) != RCSD_OK) {
        return RCSD_ERROR;
    }
    syn.cell = cell;
    syn.node = rcsd_section_node(sim, sec, x);
    syn.kind = kind;
    DYN_PUSH(sim->synapses, syn);
    site = (int) sim->synapses.n - 1;
    /* the .mod defaults */
    SP(site, RCSD_SP_TAU_RISE) = is_nmda(kind) ? 10.0 : 1.0;
    SP(site, RCSD_SP_TAU_DECAY) = is_nmda(kind) ? 35.0 : 5.0;
    SP(site, RCSD_SP_E) = (kind == RCSD_SYN_STDP_INH) ? -90.0 : 0.0;
    SP(site, RCSD_SP_MG) = 1.0;
    SP(site, RCSD_SP_KD) = 3.57;
    SP(site, RCSD_SP_GAMMA) = 0.062;
    SP(site, RCSD_SP_VSHIFT) = 0.0;
    SP(site, RCSD_SP_U) = 0.25;
    SP(site, RCSD_SP_TAU_REC) = 400.0;
    SP(site, RCSD_SP_PLASTICITY_ON) = 0.0;
    SP(site, RCSD_SP_W_INIT) = 1.0;
    SP(site, RCSD_SP_A_LTP) = 1.0;
    SP(site, RCSD_SP_A_LTD) = 1.0;
    if (kind == RCSD_SYN_STDP_INH) {
        SP(site, RCSD_SP_THETA_LTP) = -77.0;
        SP(site, RCSD_SP_THETA_LTD) = -70.0;
        SP(site, RCSD_SP_LTP_SIGMOID_HALF) = -80.0;
        SP(site, RCSD_SP_LTD_SIGMOID_HALF) = -73.0;
        SP(site, RCSD_SP_LEARNING_SLOPE) = 1.2;
    } else {
        SP(site, RCSD_SP_THETA_LTP) = -45.0;
        SP(site, RCSD_SP_THETA_LTD) = -60.0;
        SP(site, RCSD_SP_LTP_SIGMOID_HALF) = -40.0;
        SP(site, RCSD_SP_LTD_SIGMOID_HALF) = -55.0;
        SP(site, RCSD_SP_LEARNING_SLOPE) = 1.3;
    }
    SP(site, RCSD_SP_LEARNING_TAU) = 20.0;
    SP(site, RCSD_SP_W_MAX) = 5.0;
    SP(site, RCSD_SP_W_MIN) = 0.0001;
    SS(site, RCSD_SS_W) = 1.0;
    rcsd_synapse_finalize_factor(sim, site);
    return site;
}

int rcsd_synapse_count(RCSDSim* sim) {
    return (int) sim->synapses.n;
}

int rcsd_synapse_node(RCSDSim* sim, int site) {
    if (site < 0 || (size_t) site >= sim->synapses.n) {
        return RCSD_ERROR;
    }
    return sim->synapses.data[site].node;
}

double* rcsd_synapse_params(RCSDSim* sim) {
    return sim->sp;
}

double* rcsd_synapse_states(RCSDSim* sim) {
    return sim->ss;
}

int rcsd_synapse_stride(RCSDSim* sim) {
    return (int) sim->sp_cap;
}

/* the INITIAL block's clamp of tau_rise and the peak normalisation */
void rcsd_synapse_finalize_factor(RCSDSim* sim, int site) {
    double tau_rise = SP(site, RCSD_SP_TAU_RISE);
    double tau_decay = SP(site, RCSD_SP_TAU_DECAY);
    int kind = sim->synapses.data[site].kind;
    if (tau_rise / tau_decay > 0.9999) {
        tau_rise = 0.9999 * tau_decay;
    }
    if (!is_nmda(kind) && tau_rise / tau_decay < 1e-9) {
        tau_rise = tau_decay * 1e-9;
    }
    SP(site, RCSD_SP_TAU_RISE) = tau_rise;
    if (kind == RCSD_SYN_DEP) {
        if (SP(site, RCSD_SP_U) <= 0.0) {
            SP(site, RCSD_SP_U) = 1e-6;
        }
        if (SP(site, RCSD_SP_U) > 1.0) {
            SP(site, RCSD_SP_U) = 1.0;
        }
        if (SP(site, RCSD_SP_TAU_REC) <= 0.0) {
            SP(site, RCSD_SP_TAU_REC) = 1e-3;
        }
    }
    SS(site, RCSD_SS_FACTOR) = syn_factor(tau_rise, tau_decay);
    /* mod2c evaluates these every step; they only depend on dt and the taus,
     * so the cached value is the same number */
    {
        const double dt = sim->dt;
        double rate = (kind == RCSD_SYN_STDP_INH) ? -1.0 : (-1.0) / 4.0;
        SC(site, SC_EXP_RISE) = exp(dt * ((-1.0) / tau_rise));
        SC(site, SC_EXP_DECAY) = exp(dt * ((-1.0) / tau_decay));
        SC(site, SC_HALF_RISE) = exp(0.5 * dt * ((-1.0) / tau_rise));
        SC(site, SC_HALF_DECAY) = exp(0.5 * dt * ((-1.0) / tau_decay));
        SC(site, SC_EXP_LEARN) = exp(dt * rate);
    }
}

int rcsd_synapse_refresh(RCSDSim* sim, int site) {
    if (site < 0 || (size_t) site >= sim->synapses.n) {
        rcsd_set_error("no synapse %d", site);
        return RCSD_ERROR;
    }
    rcsd_synapse_finalize_factor(sim, site);
    return RCSD_OK;
}

/* INITIAL for every site and every NetCon's INITIAL block */
void rcsd_synapse_init_states(RCSDSim* sim) {
    size_t i;
    for (i = 0; i < sim->synapses.n; ++i) {
        rcsd_synapse_finalize_factor(sim, (int) i);
        SS(i, RCSD_SS_A) = 0.0;
        SS(i, RCSD_SS_B) = 0.0;
        SS(i, RCSD_SS_LEARNING_W) = 0.0;
        SS(i, RCSD_SS_LEARN_INT) = 0.0;
        SS(i, RCSD_SS_LTD) = 0.0;
        SS(i, RCSD_SS_LTP) = 0.0;
        SS(i, RCSD_SS_W) = SP(i, RCSD_SP_W_INIT);
        SS(i, RCSD_SS_G) = 0.0;
        SS(i, RCSD_SS_I) = 0.0;
    }
    for (i = 0; i < sim->connections.n; ++i) {
        int site = sim->connections.data[i].site;
        int kind = sim->synapses.data[site].kind;
        if (is_stdp(kind)) {
            W(i, 2) = SP(site, RCSD_SP_W_INIT);
            W(i, 3) = 0.0;
        } else if (kind == RCSD_SYN_DEP) {
            W(i, 2) = 1.0;
            W(i, 3) = -1e9;
        }
    }
}

/* ------------------------------------------------------------------------- */
/* inputs and connections                                                     */
/* ------------------------------------------------------------------------- */

int rcsd_add_input(RCSDSim* sim, int gid) {
    Input in;
    memset(&in, 0, sizeof in);
    in.gid = gid;
    DYN_PUSH(sim->inputs, in);
    sim->wired = 0;
    return (int) sim->inputs.n - 1;
}

static int compare_double(const void* a, const void* b) {
    double x = *(const double*) a, y = *(const double*) b;
    return (x > y) - (x < y);
}

int rcsd_set_input_spikes(RCSDSim* sim, int input, const double* times, int n) {
    Input* in;
    double* copy;
    double t;
    int i;
    if (input < 0 || (size_t) input >= sim->inputs.n) {
        rcsd_set_error("no input %d", input);
        return RCSD_ERROR;
    }
    in = &sim->inputs.data[input];
    copy = (double*) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
    if (copy == NULL) {
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    if (n > 0) {
        memcpy(copy, times, (size_t) n * sizeof(double));
        qsort(copy, (size_t) n, sizeof(double), compare_double);
    }
    free(in->times);
    in->times = copy;
    in->n = n;
    /* a train handed over mid-run starts at its first future spike */
    t = (double) sim->step * sim->dt;
    in->cursor = 0;
    for (i = 0; i < n; ++i) {
        if (copy[i] > t - 0.5 * sim->dt) {
            break;
        }
        in->cursor = i + 1;
    }
    return RCSD_OK;
}

int rcsd_add_connections(RCSDSim* sim, int n, const int* source, const int* site,
                         const double* delay, const double* weights) {
    int i;
    size_t need = sim->connections.n + (size_t) n;
    if (need > sim->w_cap) {
        size_t cap = sim->w_cap ? sim->w_cap : 256;
        double* w;
        while (cap < need) {
            cap *= 2;
        }
        w = (double*) realloc(sim->w, cap * RCSD_NWEIGHT * sizeof(double));
        if (w == NULL) {
            rcsd_set_error("out of memory");
            return RCSD_ERROR;
        }
        sim->w = w;
        sim->w_cap = cap;
    }
    for (i = 0; i < n; ++i) {
        Connection conn;
        size_t index;
        int k;
        if (site[i] < 0 || (size_t) site[i] >= sim->synapses.n) {
            rcsd_set_error("connection %d targets unknown synapse %d", i, site[i]);
            return RCSD_ERROR;
        }
        if (source[i] >= 0 && (size_t) source[i] >= sim->cells.n) {
            rcsd_set_error("connection %d has unknown source cell %d", i, source[i]);
            return RCSD_ERROR;
        }
        if (source[i] < 0 && (size_t) (-source[i] - 1) >= sim->inputs.n) {
            rcsd_set_error("connection %d has unknown input %d", i, -source[i] - 1);
            return RCSD_ERROR;
        }
        conn.source = source[i];
        conn.site = site[i];
        conn.delay = delay[i];
        conn.eff_delay = delay[i];
        index = sim->connections.n;
        DYN_PUSH(sim->connections, conn);
        for (k = 0; k < RCSD_NWEIGHT; ++k) {
            sim->w[index * RCSD_NWEIGHT + k] = weights[(size_t) i * RCSD_NWEIGHT + k];
        }
        /* the per-connection state the NetCon's INITIAL block sets, so a
         * plastic weight reads as w_init before the first initialisation */
        {
            int kind = sim->synapses.data[site[i]].kind;
            if (is_stdp(kind)) {
                sim->w[index * RCSD_NWEIGHT + 2] = SP(site[i], RCSD_SP_W_INIT);
                sim->w[index * RCSD_NWEIGHT + 3] = 0.0;
            } else if (kind == RCSD_SYN_DEP) {
                sim->w[index * RCSD_NWEIGHT + 2] = 1.0;
                sim->w[index * RCSD_NWEIGHT + 3] = -1e9;
            }
        }
    }
    sim->wired = 0;
    return RCSD_OK;
}

int rcsd_connection_count(RCSDSim* sim) {
    return (int) sim->connections.n;
}

double* rcsd_connection_weights(RCSDSim* sim) {
    return sim->w;
}

/* outgoing lists per cell and per input, delays floored at 2 dt, ring size */
int rcsd_wire(RCSDSim* sim) {
    size_t n_cells = sim->cells.n, n_inputs = sim->inputs.n, i;
    int* out_start;
    int* in_start;
    int* out_conn;
    int* in_conn;
    int* out_fill;
    int* in_fill;
    double floor_delay = 2.0 * sim->dt;
    double max_delay = 0.0;
    int n_slots, s;

    for (i = 0; i < sim->connections.n; ++i) {
        Connection* c = &sim->connections.data[i];
        c->eff_delay = c->delay > floor_delay ? c->delay : floor_delay;
        if (c->eff_delay > max_delay) {
            max_delay = c->eff_delay;
        }
    }
    sim->max_delay = max_delay;

    out_start = (int*) calloc(n_cells + 1, sizeof(int));
    in_start = (int*) calloc(n_inputs + 1, sizeof(int));
    out_conn = (int*) malloc((sim->connections.n + 1) * sizeof(int));
    in_conn = (int*) malloc((sim->connections.n + 1) * sizeof(int));
    out_fill = (int*) calloc(n_cells + 1, sizeof(int));
    in_fill = (int*) calloc(n_inputs + 1, sizeof(int));
    if (!out_start || !in_start || !out_conn || !in_conn || !out_fill || !in_fill) {
        free(out_start);
        free(in_start);
        free(out_conn);
        free(in_conn);
        free(out_fill);
        free(in_fill);
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    for (i = 0; i < sim->connections.n; ++i) {
        int src = sim->connections.data[i].source;
        if (src >= 0) {
            out_start[src + 1] += 1;
        } else {
            in_start[(-src - 1) + 1] += 1;
        }
    }
    for (i = 0; i < n_cells; ++i) {
        out_start[i + 1] += out_start[i];
    }
    for (i = 0; i < n_inputs; ++i) {
        in_start[i + 1] += in_start[i];
    }
    for (i = 0; i < sim->connections.n; ++i) {
        int src = sim->connections.data[i].source;
        if (src >= 0) {
            out_conn[out_start[src] + out_fill[src]++] = (int) i;
        } else {
            int in = -src - 1;
            in_conn[in_start[in] + in_fill[in]++] = (int) i;
        }
    }
    free(out_fill);
    free(in_fill);
    free(sim->out_start);
    free(sim->in_start);
    free(sim->out_conn);
    free(sim->in_conn);
    sim->out_start = out_start;
    sim->in_start = in_start;
    sim->out_conn = out_conn;
    sim->in_conn = in_conn;

    /* the ring: a bucket per step over the longest delay, with room for
     * the half-step rounding and the spike's own step */
    n_slots = (int) ceil(max_delay / sim->dt) + 4;
    if (sim->buckets) {
        for (s = 0; s < sim->n_slots; ++s) {
            free(sim->buckets[s].data);
        }
        free(sim->buckets);
    }
    sim->buckets = (EventList*) calloc((size_t) n_slots, sizeof(EventList));
    if (sim->buckets == NULL) {
        rcsd_set_error("out of memory");
        return RCSD_ERROR;
    }
    sim->n_slots = n_slots;
    sim->wired = 1;
    return RCSD_OK;
}

/* ------------------------------------------------------------------------- */
/* events                                                                     */
/* ------------------------------------------------------------------------- */

void rcsd_events_clear(RCSDSim* sim) {
    int s;
    if (sim->buckets == NULL) {
        return;
    }
    for (s = 0; s < sim->n_slots; ++s) {
        sim->buckets[s].n = 0;
    }
}

/* delivered at the first step whose midpoint reaches the event */
int rcsd_events_enqueue(RCSDSim* sim, int conn, double te) {
    long target = (long) ceil(te / sim->dt - 0.5);
    Event ev;
    EventList* bucket;
    if (target <= sim->step) {
        target = sim->step + 1;
    }
    if (target - sim->step >= sim->n_slots) {
        rcsd_set_error("event %g ms ahead does not fit the %d-slot ring",
                       (double) (target - sim->step) * sim->dt, sim->n_slots);
        return RCSD_ERROR;
    }
    ev.conn = conn;
    ev.te = te;
    bucket = &sim->buckets[target % sim->n_slots];
    DYN_PUSH(*bucket, ev);
    return RCSD_OK;
}

static int compare_event(const void* a, const void* b) {
    const Event* x = (const Event*) a;
    const Event* y = (const Event*) b;
    if (x->te < y->te) return -1;
    if (x->te > y->te) return 1;
    return (x->conn > y->conn) - (x->conn < y->conn);
}

int rcsd_events_deliver(RCSDSim* sim, long step) {
    EventList* bucket;
    size_t i;
    if (sim->buckets == NULL) {
        return RCSD_OK;
    }
    bucket = &sim->buckets[step % sim->n_slots];
    if (bucket->n > 1) {
        qsort(bucket->data, bucket->n, sizeof(Event), compare_event);
    }
    for (i = 0; i < bucket->n; ++i) {
        rcsd_synapse_receive(sim, bucket->data[i].conn, bucket->data[i].te);
    }
    bucket->n = 0;
    return RCSD_OK;
}

/* ------------------------------------------------------------------------- */
/* NET_RECEIVE                                                                */
/* ------------------------------------------------------------------------- */

void rcsd_synapse_receive(RCSDSim* sim, int conn, double te) {
    Connection* c = &sim->connections.data[conn];
    int site = c->site;
    int kind = sim->synapses.data[site].kind;
    double factor = SS(site, RCSD_SS_FACTOR);
    double weight = W(conn, 0), g_unit = W(conn, 1);
    double inc, primary, tau_rise, tau_decay;

    if (kind == RCSD_SYN_DEP) {
        double R = W(conn, 2), tlast = W(conn, 3), U = SP(site, RCSD_SP_U);
        R = 1.0 - (1.0 - R) * exp(-(te - tlast) / SP(site, RCSD_SP_TAU_REC));
        tlast = te;
        inc = weight * g_unit * R * U * factor;
        R = R - R * U;
        W(conn, 2) = R;
        W(conn, 3) = tlast;
    } else if (is_stdp(kind)) {
        double w_plastic = W(conn, 2);
        if (SP(site, RCSD_SP_PLASTICITY_ON) > 0.5) {
            double delta_learn = SS(site, RCSD_SS_LEARN_INT) - W(conn, 3);
            w_plastic = w_plastic + delta_learn * w_plastic;
            W(conn, 3) = SS(site, RCSD_SS_LEARN_INT);
            if (w_plastic > SP(site, RCSD_SP_W_MAX)) {
                w_plastic = SP(site, RCSD_SP_W_MAX);
            }
            if (w_plastic < SP(site, RCSD_SP_W_MIN)) {
                w_plastic = SP(site, RCSD_SP_W_MIN);
            }
            W(conn, 2) = w_plastic;
            SS(site, RCSD_SS_W) = w_plastic;
        }
        inc = w_plastic * weight * g_unit * factor;
    } else {
        inc = weight * g_unit * factor;
    }

    /* nrn_netrec_state_adjust: the increment is what the staggered scheme
     * would have had half a step later */
    tau_rise = SP(site, RCSD_SP_TAU_RISE);
    tau_decay = SP(site, RCSD_SP_TAU_DECAY);
    primary = (SS(site, RCSD_SS_A) + inc) - SS(site, RCSD_SS_A);
    primary += (1.0 - SC(site, SC_HALF_RISE)) * (-(0.0) / ((-1.0) / tau_rise) - primary);
    SS(site, RCSD_SS_A) += primary;
    primary = (SS(site, RCSD_SS_B) + inc) - SS(site, RCSD_SS_B);
    primary += (1.0 - SC(site, SC_HALF_DECAY)) * (-(0.0) / ((-1.0) / tau_decay) - primary);
    SS(site, RCSD_SS_B) += primary;
}

/* ------------------------------------------------------------------------- */
/* BREAKPOINT and the state update                                            */
/* ------------------------------------------------------------------------- */

/* the BREAKPOINT body of one site at voltage v; the STDP learning signal is
 * accumulated here because NEURON runs it inside `_nrn_current` too */
static double site_current(RCSDSim* sim, int site, int kind, double v) {
    double g = SS(site, RCSD_SS_B) - SS(site, RCSD_SS_A);
    double i;
    if (is_nmda(kind)) {
        double pnmda = mgblock(v, SP(site, RCSD_SP_GAMMA), SP(site, RCSD_SP_VSHIFT),
                               SP(site, RCSD_SP_MG), SP(site, RCSD_SP_KD));
        i = g * pnmda * (v - SP(site, RCSD_SP_E));
    } else {
        i = g * (v - SP(site, RCSD_SP_E));
    }
    if (is_stdp(kind) && SP(site, RCSD_SP_PLASTICITY_ON) > 0.0) {
        double slope = SP(site, RCSD_SP_LEARNING_SLOPE);
        double ltd, ltp;
        if (kind == RCSD_SYN_STDP_INH) {
            ltd = (v - SP(site, RCSD_SP_THETA_LTD) < 0.0)
                      ? sigmoid_thr(slope, v, SP(site, RCSD_SP_LTD_SIGMOID_HALF))
                      : 0.0;
            ltp = (v - SP(site, RCSD_SP_THETA_LTP) < 0.0)
                      ? sigmoid_thr(slope, v, SP(site, RCSD_SP_LTP_SIGMOID_HALF))
                      : 0.0;
        } else {
            ltd = (v - SP(site, RCSD_SP_THETA_LTD) > 0.0)
                      ? sigmoid_thr(slope, v, SP(site, RCSD_SP_LTD_SIGMOID_HALF))
                      : 0.0;
            ltp = (v - SP(site, RCSD_SP_THETA_LTP) > 0.0)
                      ? sigmoid_thr(slope, v, SP(site, RCSD_SP_LTP_SIGMOID_HALF))
                      : 0.0;
        }
        SS(site, RCSD_SS_LTD) = ltd;
        SS(site, RCSD_SS_LTP) = ltp;
        SS(site, RCSD_SS_LEARNING_W) =
            SS(site, RCSD_SS_LEARNING_W) +
            sigmoid_sat(slope, (-SP(site, RCSD_SP_A_LTD) * ltd + SP(site, RCSD_SP_A_LTP) * 2.0 * ltp) /
                                   SP(site, RCSD_SP_LEARNING_TAU)) /
                5000.0;
    }
    SS(site, RCSD_SS_G) = g;
    return i;
}

void rcsd_synapse_currents(RCSDSim* sim) {
    size_t s;
    for (s = 0; s < sim->synapses.n; ++s) {
        Synapse* syn = &sim->synapses.data[s];
        int node = syn->node;
        double v = sim->v[node];
        double i1 = site_current(sim, (int) s, syn->kind, v + 0.001);
        double i0 = site_current(sim, (int) s, syn->kind, v);
        double g = (i1 - i0) / 0.001;
        double scale = 1e2 / sim->area[node];
        SS(s, RCSD_SS_I) = i0;
        sim->rhs[node] -= i0 * scale;
        sim->d[node] += g * scale;
    }
}

void rcsd_synapse_state_step(RCSDSim* sim) {
    const double dt = sim->dt;
    size_t s;
    for (s = 0; s < sim->synapses.n; ++s) {
        int kind = sim->synapses.data[s].kind;
        /* mod2c's `-(0.0)/((-1.0)/tau) - A` is `+0.0 - A` for any positive tau */
        SS(s, RCSD_SS_A) = SS(s, RCSD_SS_A) + (1.0 - SC(s, SC_EXP_RISE)) * (0.0 - SS(s, RCSD_SS_A));
        SS(s, RCSD_SS_B) = SS(s, RCSD_SS_B) + (1.0 - SC(s, SC_EXP_DECAY)) * (0.0 - SS(s, RCSD_SS_B));
        if (is_stdp(kind)) {
            double rate = (kind == RCSD_SYN_STDP_INH) ? -1.0 : (-1.0) / 4.0;
            double lw = SS(s, RCSD_SS_LEARNING_W);
            lw = lw + (1.0 - SC(s, SC_EXP_LEARN)) * (-(0.0) / rate - lw);
            SS(s, RCSD_SS_LEARNING_W) = lw;
            SS(s, RCSD_SS_LEARN_INT) = SS(s, RCSD_SS_LEARN_INT) - dt * (-(lw));
        }
    }
}
