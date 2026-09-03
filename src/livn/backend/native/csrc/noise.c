/* Gfluct3: fluctuating excitatory and inhibitory conductances, one point
 * process per section, advanced with the exact Ornstein-Uhlenbeck update on
 * a Random123 stream that restarts with every initialisation.
 */
#include <math.h>
#include <string.h>

#include "internal.h"

static Noise* find_noise(RCSDSim* sim, int cell, int section) {
    size_t i;
    for (i = 0; i < sim->noise.n; ++i) {
        Noise* n = &sim->noise.data[i];
        if (n->cell == cell && n->section == section) {
            return n;
        }
    }
    return NULL;
}

static double exptrap(double x) {
    if (x >= 700.0) {
        return exp(700.0);
    }
    return exp(x);
}

/* the INITIAL block's derived quantities, as neuron_noise_configure recomputes them */
static void configure(Noise* n) {
    if (n->tau_e > 0.0) {
        n->exp_e = exp(-n->h / n->tau_e);
        n->amp_e = n->std_e * sqrt(fmax(0.0, 1.0 - exp(-2.0 * n->h / n->tau_e)));
    } else {
        n->exp_e = 0.0;
        n->amp_e = 0.0;
    }
    if (n->tau_i > 0.0) {
        n->exp_i = exp(-n->h / n->tau_i);
        n->amp_i = n->std_i * sqrt(fmax(0.0, 1.0 - exp(-2.0 * n->h / n->tau_i)));
    } else {
        n->exp_i = 0.0;
        n->amp_i = 0.0;
    }
}

int rcsd_set_noise(RCSDSim* sim, int cell, int section, double g_e0, double g_i0, double std_e,
                   double std_i, double tau_e, double tau_i, double E_e, double E_i, double h,
                   int on) {
    Noise* n = find_noise(sim, cell, section);
    int sec = rcsd_cell_section(sim, cell, section);
    if (sec < 0) {
        rcsd_set_error("cell %d has no section %d", cell, section);
        return RCSD_ERROR;
    }
    if (n == NULL) {
        Noise fresh;
        memset(&fresh, 0, sizeof fresh);
        fresh.cell = cell;
        fresh.section = section;
        fresh.node = rcsd_section_node(sim, sec, 0.5);
        fresh.t_last = 0.0;
        DYN_PUSH(sim->noise, fresh);
        n = &sim->noise.data[sim->noise.n - 1];
        r123_seed(&n->stream, 0, 0, 0);
    }
    n->g_e0 = g_e0;
    n->g_i0 = g_i0;
    n->std_e = std_e;
    n->std_i = std_i;
    n->tau_e = tau_e;
    n->tau_i = tau_i;
    n->E_e = E_e;
    n->E_i = E_i;
    n->h = h;
    n->on = on;
    configure(n);
    return RCSD_OK;
}

int rcsd_set_noise_stream(RCSDSim* sim, int cell, int section, unsigned id1, unsigned id2,
                          unsigned id3) {
    Noise* n = find_noise(sim, cell, section);
    if (n == NULL) {
        rcsd_set_error("cell %d section %d carries no noise yet; set it first", cell, section);
        return RCSD_ERROR;
    }
    n->id1 = id1;
    n->id2 = id2;
    n->id3 = id3;
    n->seeded = 1;
    r123_seed(&n->stream, id1, id2, id3);
    return RCSD_OK;
}

int rcsd_noise_count(RCSDSim* sim) {
    return (int) sim->noise.n;
}

void rcsd_noise_init(RCSDSim* sim) {
    size_t i;
    for (i = 0; i < sim->noise.n; ++i) {
        Noise* n = &sim->noise.data[i];
        r123_setseq(&n->stream, 0, 0);
        n->g_e1 = 0.0;
        n->g_i1 = 0.0;
        n->ival = 0.0;
        /* INITIAL recomputes from h, with exptrap */
        if (n->tau_e != 0.0) {
            n->exp_e = exp(-n->h / n->tau_e);
            n->amp_e = n->std_e * sqrt(1.0 - exptrap(-2.0 * n->h / n->tau_e));
        }
        if (n->tau_i != 0.0) {
            n->exp_i = exp(-n->h / n->tau_i);
            n->amp_i = n->std_i * sqrt(1.0 - exptrap(-2.0 * n->h / n->tau_i));
        }
        n->t_last = 0.0;
    }
}

/* BEFORE BREAKPOINT at the step midpoint, then the current of the
 * conductances clipped at zero, in the matrix like every other conductance */
void rcsd_noise_advance(RCSDSim* sim, double t_mid) {
    size_t i;
    for (i = 0; i < sim->noise.n; ++i) {
        Noise* n = &sim->noise.data[i];
        if ((n->tau_e != 0.0) || (n->tau_i != 0.0)) {
            if (t_mid - n->t_last >= n->h - 1e-9) {
                /* A site with both amplitudes at zero (Gfluct3 on every axon
                 * section, say) multiplies its deviates by zero; its stream is
                 * its own, so the draws are skipped without touching any
                 * other site. The one thing that differs from NEURON is the
                 * position of this stream if its amplitude is raised
                 * mid-run, which is then a different realisation, not a
                 * biased one. */
                if (n->amp_e == 0.0 && n->amp_i == 0.0) {
                    if (n->tau_e != 0.0) {
                        n->g_e1 = n->exp_e * n->g_e1;
                    }
                    if (n->tau_i != 0.0) {
                        n->g_i1 = n->exp_i * n->g_i1;
                    }
                    n->t_last = t_mid;
                    continue;
                }
                if (n->tau_e != 0.0) {
                    n->g_e1 = n->exp_e * n->g_e1 + n->amp_e * r123_normal(&n->stream);
                }
                if (n->tau_i != 0.0) {
                    n->g_i1 = n->exp_i * n->g_i1 + n->amp_i * r123_normal(&n->stream);
                }
                n->t_last = t_mid;
            }
        }
    }
}

/* invoked from the current evaluation through the synapse pass */
void rcsd_noise_currents(RCSDSim* sim);
void rcsd_noise_currents(RCSDSim* sim) {
    size_t i;
    for (i = 0; i < sim->noise.n; ++i) {
        Noise* n = &sim->noise.data[i];
        double v = sim->v[n->node];
        double g_e, g_i, i0, i1, g, scale;
        if (n->on <= 0) {
            n->ival = 0.0;
            continue;
        }
        g_e = n->g_e0 + n->g_e1;
        if (g_e < 0.0) {
            g_e = 0.0;
        }
        g_i = n->g_i0 + n->g_i1;
        if (g_i < 0.0) {
            g_i = 0.0;
        }
        n->g_e = g_e;
        n->g_i = g_i;
        i1 = g_e * ((v + 0.001) - n->E_e) + g_i * ((v + 0.001) - n->E_i);
        i0 = g_e * (v - n->E_e) + g_i * (v - n->E_i);
        n->ival = i0;
        g = (i1 - i0) / 0.001;
        scale = 1e2 / sim->area[n->node];
        sim->rhs[n->node] -= i0 * scale;
        sim->d[n->node] += g * scale;
    }
}
