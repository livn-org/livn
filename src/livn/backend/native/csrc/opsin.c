/* RhO3c: the three-state rhodopsin (Nikolic et al. 2009) as a point process,
 * with the kinetic scheme advanced by implicit Euler under the conservation
 * constraint, which is what NEURON's `METHOD sparse` does.
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
        sim->rhs[op->node] -= i0 * scale;
        sim->d[op->node] += g * scale;
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
