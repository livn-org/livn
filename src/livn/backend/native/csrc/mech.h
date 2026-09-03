/* The channel, ion and opsin equations, lifted from the .mod files.
 *
 * Every function is a pure function of its arguments so it can be checked
 * against a reference value in isolation. Rates are what NEURON evaluates in
 * `rates(v)` from the DERIVATIVE block; the BREAKPOINT bodies are separate
 * so the current can be evaluated at `v + 0.001` for the numerical Jacobian
 * exactly as NEURON does.
 */
#ifndef RCSD_MECH_H
#define RCSD_MECH_H

#include <math.h>

/* --- Nas.mod ---------------------------------------------------------------- */
static inline double nas_minf(double v, double vhalf, double slope) {
    return 1.0 / (1.0 + exp(-(v - vhalf) / slope));
}
static inline double nas_htau(double v) {
    return 30.0 / (exp((v + 50.0) / 15.0) + exp(-(v + 50.0) / 16.0));
}
static inline double nas_hinf(double v) {
    return 1.0 / (1.0 + exp((v + 55.0) / 7.0));
}
static inline double nas_current(double v, double gmax, double minf, double h, double ena) {
    double g = gmax * minf * minf * minf * h;
    return g * (v - ena);
}

/* --- Kdr.mod ---------------------------------------------------------------- */
static inline double kdr_ntau(double v) {
    return 7.0 / (exp((v + 40.0) / 40.0) + exp(-(v + 40.0) / 50.0));
}
static inline double kdr_ninf(double v) {
    return 1.0 / ((exp(-(v + 28.0) / 15.0)) + 1.0);
}
static inline double kdr_current(double v, double gmax, double n, double ek) {
    double g = gmax * n * n * n * n;
    return g * (v - ek);
}

/* --- GHK, as in CaN.mod and CaL.mod ------------------------------------------- */
static inline double ghk_efun(double z) {
    if (fabs(z) < 1e-4) {
        return 1.0 - z / 2.0;
    }
    return z / (exp(z) - 1.0);
}
/* f = KTF(celsius)/2; CaN uses 36/293.15 and CaL 25/293.15 in KTF */
static inline double ghk_driving(double v, double ci, double co, double f) {
    double nu = v / f;
    return -f * (1.0 - (ci / co) * exp(nu)) * ghk_efun(nu);
}
static inline double can_f(double celsius) {
    return ((36.0 / 293.15) * (celsius + 273.15)) / 2.0;
}
static inline double cal_f(double celsius) {
    return ((25.0 / 293.15) * (celsius + 273.15)) / 2.0;
}

/* --- CaN.mod ---------------------------------------------------------------- */
#define CAN_MTAU 4.0
#define CAN_HTAU 40.0
static inline double can_minf(double v) {
    return 1.0 / (1.0 + exp((v + 30.0) / -5.0));
}
static inline double can_hinf(double v) {
    return 1.0 / (1.0 + exp((v + 45.0) / 5.0));
}
static inline double can_current(double v, double gmax, double m, double h, double cai,
                                 double cao, double f) {
    double g = gmax * m * m * h;
    return g * ghk_driving(v, cai, cao, f);
}

/* --- CaL.mod ---------------------------------------------------------------- */
#define CAL_MTAU 60.0
static inline double cal_minf(double v) {
    return 1.0 / (1.0 + exp((v + 40.0) / -7.0));
}
static inline double cal_current(double v, double gmax, double m, double cai, double cao,
                                 double f) {
    double g = gmax * m;
    return g * ghk_driving(v, cai, cao, f);
}

/* --- KCa.mod ---------------------------------------------------------------- */
static inline double kca_current(double v, double gmax, double Kd, double cai, double ek) {
    double g = gmax * (cai / (cai + Kd));
    return g * (v - ek);
}

/* --- Ka_v1in.mod -------------------------------------------------------------- */
#define KA_ATAU 1.0
#define KA_BTAU 15.0
static inline double ka_ainf(double v) {
    return 1.0 / (1.0 + exp(-(v + 36.0) / 8.0));
}
static inline double ka_binf(double v) {
    return 1.0 / (1.0 + exp((v + 66.0) / 8.0));
}
static inline double ka_current(double v, double gmax, double a, double b, double ek) {
    double g = gmax * a * b;
    return g * (v - ek);
}

/* --- cnexp: y' = (yinf - y)/tau over one step --------------------------------- */
static inline double cnexp_relax(double y, double yinf, double tau, double dt) {
    return y + (1.0 - exp(dt * (-1.0 / tau))) * (yinf - y);
}

/* --- Na_conc.mod / K_conc.mod: x' = -i/(2 F d) 1e4 - beta (x - x0) ---------------- */
static inline double conc_step(double x, double x0, double flux, double beta, double d,
                               double dt, double faraday) {
    /* mod2c's cnexp form, kept literally so the rounding matches */
    double a = -(beta) * (1.0);
    double b = ((flux) / (2.0 * faraday * d)) * (1e4) - (beta) * ((-x0));
    return x + (1.0 - exp(dt * a)) * (-(b) / (a) -x);
}

/* --- Ca_conc.mod --------------------------------------------------------------- */
static inline double ca_conc_step(double cai, double cai0, double ica, double irest, double f,
                                  double alpha, double kCa, double dt) {
    double channel_flow = -alpha * (ica - irest);
    double a, b;
    if (channel_flow < 0.0) {
        channel_flow = 0.0;
    }
    a = (f) * ((-(kCa) * (1.0)));
    b = (f) * ((channel_flow - (kCa) * ((-cai0))));
    return cai + (1.0 - exp(dt * a)) * (-(b) / (a) -cai);
}

/* --- Nernst, as nrn_nernst -------------------------------------------------------- */
static inline double nernst(double ci, double co, double z, double celsius, double R,
                            double F) {
    double ktf = 1000.0 * R * (celsius + 273.15) / F;
    if (ci <= 0.0) {
        return 1e6;
    }
    if (co <= 0.0) {
        return -1e6;
    }
    return ktf / z * log(co / ci);
}

/* --- synapse helpers -------------------------------------------------------------- */
static inline double syn_factor(double tau_rise, double tau_decay) {
    double tp = (tau_rise * tau_decay) / (tau_decay - tau_rise) * log(tau_decay / tau_rise);
    double factor = -exp(-tp / tau_rise) + exp(-tp / tau_decay);
    return 1.0 / factor;
}
static inline double mgblock(double v, double gamma, double vshift, double mg, double Kd) {
    return 1.0 / (1.0 + exp(gamma * -(v + vshift)) * (mg / Kd));
}
static inline double sigmoid_thr(double slope, double value, double thr) {
    return 1.0 / (1.0 + pow(slope, -(value - thr)));
}
static inline double sigmoid_sat(double slope, double value) {
    return 2.0 / (1.0 + pow(slope, -value)) - 1.0;
}

/* --- RhO3c.mod -------------------------------------------------------------------- */
static inline void rho3c_rates(double phi, double k_a, double k_r, double p, double q,
                               double Gr0, double phi_m, double* Ga, double* Gr) {
    if (phi > 0.0) {
        *Ga = k_a * 1.0 / (1.0 + pow(phi_m, p) / pow(phi, p));
        *Gr = Gr0 + k_r * 1.0 / (1.0 + pow(phi_m, q) / pow(phi, q));
    } else {
        *Ga = 0.0;
        *Gr = Gr0;
    }
}
static inline double rho3c_v1(double E, double v0) {
    return (70.0 + E) / (exp((70.0 + E) / v0) - 1.0);
}
static inline double rho3c_current(double v, double O, double g0, double E, double v0,
                                   double v1) {
    double dv = v - E;
    double fv;
    if (fabs(dv) < 1e-9) {
        fv = v1 / v0;
    } else {
        fv = (1.0 - exp(-dv / v0)) * v1 / dv;
    }
    return g0 * O * fv * dv * (1e-6);
}

#endif
