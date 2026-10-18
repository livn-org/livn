/* RhO3c: the three-state rhodopsin (Nikolic et al. 2009) as a point process,
 * with the kinetic scheme advanced by implicit Euler under the conservation
 * constraint, which is what NEURON's `METHOD sparse` does.
 *
 * ChR2H134R: the four-state channelrhodopsin-2 (Williams et al. 2013) as a
 * density mechanism, one instance per segment, advanced by mod2c's cnexp.
 */
#include <math.h>
#include <string.h>

#include "internal.h"
#include "mech.h"

int rcsd_add_opsin(RCSDSim* sim, int cell, int section, double x) {
    Opsin op;
    int sec = rcsd_cell_section(sim, cell, section);
    if (sec < 0) {
        rcsd_set_error("cell %d has no section %d", cell, section);
        return RCSD_ERROR;
    }
    memset(&op, 0, sizeof op);
    op.cell = cell;
    op.section = section;
    op.node = rcsd_section_node(sim, sec, x);
    op.g0 = 1.0;
    op.E = 0.0;
    op.v0 = 43.0;
    op.k_a = 0.28;
    op.k_r = 0.28;
    op.p = 0.4;
    op.q = 0.4;
    op.Gd = 0.0909;
    op.Gr0 = 0.0002;
    op.phi_m = 1e16;
    op.v1 = rho3c_v1(op.E, op.v0);
    op.C = 1.0;
    op.O = 0.0;
    op.phi = 0.0;
    DYN_PUSH(sim->opsins, op);
    sim->pp_dirty = 1;
    if (sim->cells.data[cell].opsin < 0) {
        sim->cells.data[cell].opsin = (int) sim->opsins.n - 1;
    }
    return (int) sim->opsins.n - 1;
}

int rcsd_opsin_set(RCSDSim* sim, int opsin, double g0, double E, double v0, double k_a,
                   double k_r, double p, double q, double Gd, double Gr0, double phi_m) {
    Opsin* op;
    if (opsin < 0 || (size_t) opsin >= sim->opsins.n) {
        rcsd_set_error("no opsin %d", opsin);
        return RCSD_ERROR;
    }
    op = &sim->opsins.data[opsin];
    op->g0 = g0;
    op->E = E;
    op->v0 = v0;
    op->k_a = k_a;
    op->k_r = k_r;
    op->p = p;
    op->q = q;
    op->Gd = Gd;
    op->Gr0 = Gr0;
    op->phi_m = phi_m;
    op->v1 = rho3c_v1(E, v0);
    return RCSD_OK;
}

int rcsd_opsin_count(RCSDSim* sim) {
    return (int) sim->opsins.n;
}

int rcsd_opsin_state(RCSDSim* sim, int opsin, double* C, double* O, double* phi) {
    Opsin* op;
    if (opsin < 0 || (size_t) opsin >= sim->opsins.n) {
        rcsd_set_error("no opsin %d", opsin);
        return RCSD_ERROR;
    }
    op = &sim->opsins.data[opsin];
    if (C) *C = op->C;
    if (O) *O = op->O;
    if (phi) *phi = op->phi;
    return RCSD_OK;
}

void rcsd_opsin_init(RCSDSim* sim) {
    size_t i;
    for (i = 0; i < sim->opsins.n; ++i) {
        Opsin* op = &sim->opsins.data[i];
        op->C = 1.0;
        op->O = 0.0;
        op->phi = 0.0;
        op->cur = 0.0;
        op->dcur = 0.0;
        op->v1 = rho3c_v1(op->E, op->v0);
    }
}

void rcsd_opsin_currents(RCSDSim* sim);
void rcsd_opsin_currents(RCSDSim* sim) {
    size_t i;
    for (i = 0; i < sim->opsins.n; ++i) {
        Opsin* op = &sim->opsins.data[i];
        double v = sim->v[op->node];
        double i1 = rho3c_current(v + 0.001, op->O, op->g0, op->E, op->v0, op->v1);
        double i0 = rho3c_current(v, op->O, op->g0, op->E, op->v0, op->v1);
        double g = (i1 - i0) / 0.001;
        double scale = 1e2 / sim->area[op->node];
        op->cur = i0 * scale;
        op->dcur = g * scale;
    }
}

/* implicit Euler on (O, D) with C = 1 - O - D */
void rcsd_opsin_state_step(RCSDSim* sim) {
    const double dt = sim->dt;
    size_t i;
    for (i = 0; i < sim->opsins.n; ++i) {
        Opsin* op = &sim->opsins.data[i];
        double Ga, Gr;
        double O0 = op->O, D0 = 1.0 - op->C - op->O;
        double a11, a12, a21, a22, b1, b2, det, O1, D1;
        rho3c_rates(op->phi, op->k_a, op->k_r, op->p, op->q, op->Gr0, op->phi_m, &Ga, &Gr);
        /* O' = Ga C - Gd O = Ga (1 - O - D) - Gd O
         * D' = Gd O - Gr D
         * (1 + dt (Ga + Gd)) O + dt Ga D = O0 + dt Ga
         * -dt Gd O + (1 + dt Gr) D = D0                     */
        a11 = 1.0 + dt * (Ga + op->Gd);
        a12 = dt * Ga;
        a21 = -dt * op->Gd;
        a22 = 1.0 + dt * Gr;
        b1 = O0 + dt * Ga;
        b2 = D0;
        det = a11 * a22 - a12 * a21;
        O1 = (b1 * a22 - a12 * b2) / det;
        D1 = (a11 * b2 - a21 * b1) / det;
        op->O = O1;
        op->C = 1.0 - O1 - D1;
    }
}

/* --- ChR2H134R ----------------------------------------------------------------- */
/* the defaults NEURON holds, which nocmodl prints with %g: hc_J_m is the
 * .mod's 1.986446e-25 to six digits */
static const double chr2_defaults[RCSD_CHR2_NPARAM] = {
    4e-4,         /* gmax, S/cm2 */
    0.0,          /* EChR2 */
    0.1,          /* gamma */
    10.6408,      /* A */
    -14.6408,     /* B */
    -42.7671,     /* C */
    470.0,        /* wavelength */
    1.98645e-25,  /* hc_J_m */
    1.3,          /* wloss */
    12e-20,       /* sigma_retinal_m2 */
    1.3,          /* tauChR2 */
    22.0,         /* temp */
    1.97,         /* Q10_Gd1 */
    1.77,         /* Q10_Gd2 */
    2.56,         /* Q10_Gr */
    1.1,          /* Q10_e12dark */
    1.95,         /* Q10_e21dark */
    1.46,         /* Q10_epsilon1 */
    2.77,         /* Q10_epsilon2 */
};

int rcsd_add_chr2(RCSDSim* sim, int cell, int section, double x) {
    ChR2 ch;
    int sec = rcsd_cell_section(sim, cell, section);
    if (sec < 0) {
        rcsd_set_error("cell %d has no section %d", cell, section);
        return RCSD_ERROR;
    }
    memset(&ch, 0, sizeof ch);
    ch.cell = cell;
    ch.section = section;
    ch.node = rcsd_section_node(sim, sec, x);
    memcpy(ch.param, chr2_defaults, sizeof ch.param);
    ch.C1 = 1.0;
    DYN_PUSH(sim->chr2, ch);
    sim->pp_dirty = 1;
    return (int) sim->chr2.n - 1;
}

static ChR2* chr2_at(RCSDSim* sim, int index) {
    if (index < 0 || (size_t) index >= sim->chr2.n) {
        rcsd_set_error("no ChR2H134R %d", index);
        return NULL;
    }
    return &sim->chr2.data[index];
}

int rcsd_chr2_set(RCSDSim* sim, int index, int param, double value) {
    ChR2* ch = chr2_at(sim, index);
    if (ch == NULL) {
        return RCSD_ERROR;
    }
    if (param < 0 || param >= RCSD_CHR2_NPARAM) {
        rcsd_set_error("no ChR2H134R parameter %d", param);
        return RCSD_ERROR;
    }
    ch->param[param] = value;
    return RCSD_OK;
}

double rcsd_chr2_get(RCSDSim* sim, int index, int param) {
    ChR2* ch = chr2_at(sim, index);
    if (ch == NULL || param < 0 || param >= RCSD_CHR2_NPARAM) {
        return NAN;
    }
    return ch->param[param];
}

int rcsd_chr2_count(RCSDSim* sim) {
    return (int) sim->chr2.n;
}

int rcsd_chr2_state(RCSDSim* sim, int index, double* out) {
    ChR2* ch = chr2_at(sim, index);
    if (ch == NULL) {
        return RCSD_ERROR;
    }
    out[0] = ch->O1;
    out[1] = ch->O2;
    out[2] = ch->C1;
    out[3] = ch->p;
    out[4] = ch->irr;
    return RCSD_OK;
}

/* INITIAL: Irradiance = 0, C1 = 1, O1 = O2 = p = 0 */
void rcsd_chr2_init(RCSDSim* sim) {
    size_t i;
    for (i = 0; i < sim->chr2.n; ++i) {
        ChR2* ch = &sim->chr2.data[i];
        ch->irr = 0.0;
        ch->C1 = 1.0;
        ch->O1 = 0.0;
        ch->O2 = 0.0;
        ch->p = 0.0;
        ch->cur = 0.0;
        ch->dcur = 0.0;
    }
}

/* nrn_cur: the current density and its numerical conductance */
void rcsd_chr2_currents(RCSDSim* sim) {
    size_t i;
    for (i = 0; i < sim->chr2.n; ++i) {
        ChR2* ch = &sim->chr2.data[i];
        double v = sim->v[ch->node];
        double i1 = chr2_current(v + .001, ch->param, ch->O1, ch->O2);
        double i0 = chr2_current(v, ch->param, ch->O1, ch->O2);
        ch->cur = i0;
        ch->dcur = (i1 - i0) / .001;
    }
}

/* nrn_state: rates at the new voltage and this step's irradiance */
void rcsd_chr2_state_step(RCSDSim* sim) {
    size_t i;
    for (i = 0; i < sim->chr2.n; ++i) {
        ChR2* ch = &sim->chr2.data[i];
        ChR2Rates r;
        chr2_rates(sim->v[ch->node], ch->irr, sim->celsius, ch->param, &r);
        chr2_states(&ch->O1, &ch->O2, &ch->C1, &ch->p, &r, ch->param[RCSD_CHR2_TAU], sim->dt);
    }
}
