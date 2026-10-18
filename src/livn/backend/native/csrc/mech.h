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

#include "rcsd.h"

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
/* mod2c renders minf^3 as pow(minf, 3.0), which rounds unlike minf*minf*minf;
 * the caller evaluates it once for both the current and its derivative */
static inline double nas_m3(double minf) {
    return pow(minf, 3.0);
}
static inline double nas_current(double v, double gmax, double m3, double h, double ena) {
    double g = gmax * m3 * h;
    return g * (v - ena);
}
/* optional slow inactivation, Nas.mod's s_on branch: g = gmax * minf^3 * h * s,
 * multiplied in the same order mod2c evaluates it */
static inline double nas_current_s(double v, double gmax, double m3, double h, double s,
                                   double ena) {
    double g = gmax * m3 * h * s;
    return g * (v - ena);
}
/* Nas.mod rates(): alpha_s recovery, beta_s entry (reconstructed sigmoid) */
static inline double nas_s_alpha(double v, double speed) {
    return speed * 0.001 * exp(-(v + 85.0) / 30.0);
}
static inline double nas_s_beta(double v, double speed) {
    return speed * 0.0034 / (1.0 + exp(-(v + 17.0) / 10.0));
}
static inline double nas_stau(double v, double speed) {
    return 1.0 / (nas_s_alpha(v, speed) + nas_s_beta(v, speed));
}
static inline double nas_sinf(double v, double floor, double speed) {
    double as = nas_s_alpha(v, speed);
    double bs = nas_s_beta(v, speed);
    return floor + (1.0 - floor) * as / (as + bs);
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
    /* mod2c's form, literally: the steady state is -(yinf/tau)/(-1/tau), which
     * is yinf only up to rounding */
    return y + (1.0 - exp(dt * (((-1.0)) / tau))) * (-((yinf) / tau) / (((-1.0)) / tau) - y);
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

/* --- ChR2H134R.mod (Williams et al. 2013) ----------------------------------------
 * Each expression is the one mod2c emits for the .mod: rates() at the new
 * voltage, then the four cnexp updates in order, each reading the states
 * already updated before it. P is indexed by RCSD_CHR2_*. */
typedef struct {
    double Gd1, Gd2, Gr, e12, e21, epsilon1, epsilon2, F, S0;
} ChR2Rates;

static inline void chr2_rates(double v, double irr, double celsius, const double* P,
                              ChR2Rates* r) {
    double e12dark, e21dark, logphi0, Ephoton, flux;
    const double temp = P[RCSD_CHR2_TEMP];
    e12dark = 0.011;
    e21dark = 0.008;
    r->epsilon1 = 0.8535;
    r->epsilon2 = 0.14;
    r->Gd1 = 0.075 + 0.043 * tanh(-(v + 20.0) / 20.0);
    r->Gd2 = 0.05;
    r->Gr = 0.0000434587 * exp(-0.0211539274 * v);
    e12dark = e12dark * pow(P[RCSD_CHR2_Q10_E12], ((celsius - temp) / 10.0));
    e21dark = e21dark * pow(P[RCSD_CHR2_Q10_E21], ((celsius - temp) / 10.0));
    r->epsilon1 = r->epsilon1 * pow(P[RCSD_CHR2_Q10_EPS1], ((celsius - temp) / 10.0));
    r->epsilon2 = r->epsilon2 * pow(P[RCSD_CHR2_Q10_EPS2], ((celsius - temp) / 10.0));
    r->Gd1 = r->Gd1 * pow(P[RCSD_CHR2_Q10_GD1], ((celsius - temp) / 10.0));
    r->Gd2 = r->Gd2 * pow(P[RCSD_CHR2_Q10_GD2], ((celsius - temp) / 10.0));
    r->Gr = r->Gr * pow(P[RCSD_CHR2_Q10_GR], ((celsius - temp) / 10.0));
    if (irr > 0.0) {
        logphi0 = log(1.0 + irr / 0.024);
    } else {
        logphi0 = 0.0;
    }
    r->e12 = e12dark + 0.005 * logphi0;
    r->e21 = e21dark + 0.004 * logphi0;
    r->S0 = 0.5 * (1.0 + tanh(120.0 * (100.0 * irr - 0.1)));
    Ephoton = 1e9 * P[RCSD_CHR2_HC] / P[RCSD_CHR2_WAVELENGTH];
    flux = 1000.0 * irr / Ephoton;
    r->F = flux * P[RCSD_CHR2_SIGMA] / (P[RCSD_CHR2_WLOSS] * 1000.0);
}

static inline void chr2_states(double* O1, double* O2, double* C1, double* p,
                               const ChR2Rates* r, double tauChR2, double dt) {
    const double Gd1 = r->Gd1, Gd2 = r->Gd2, Gr = r->Gr, e12 = r->e12, e21 = r->e21;
    const double epsilon1 = r->epsilon1, epsilon2 = r->epsilon2, F = r->F, S0 = r->S0;
    *O1 = *O1 + (1. - exp(dt * ((-(Gd1 + e12)) * (1.0)))) *
                    (-((e21) * (*O2) + (((epsilon1) * (F)) * (*p)) * (*C1)) /
                         ((-(Gd1 + e12)) * (1.0)) -
                     *O1);
    *O2 = *O2 + (1. - exp(dt * ((-(Gd2 + e21)) * (1.0) + (epsilon2 * F * *p) * (((-1.0)))))) *
                    (-((e12) * (*O1) + (((epsilon2) * (F)) * (*p)) * ((1.0 - *C1 - *O1))) /
                         ((-(Gd2 + e21)) * (1.0) + (((epsilon2) * (F)) * (*p)) * (((-1.0)))) -
                     *O2);
    *C1 = *C1 + (1. - exp(dt * ((-epsilon1 * F * *p) * (1.0) + (Gr) * (((-1.0)))))) *
                    (-((Gd1) * (*O1) + (Gr) * ((1.0 - *O1 - *O2))) /
                         ((((-epsilon1) * (F)) * (*p)) * (1.0) + (Gr) * (((-1.0)))) -
                     *C1);
    *p = *p + (1. - exp(dt * ((((-1.0))) / tauChR2))) *
                  (-(((S0)) / tauChR2) / ((((-1.0))) / tauChR2) - *p);
}

/* BREAKPOINT: (A + B exp(v/C)) (1 - E/v) for (A + B exp(v/C))/v (v - E), with
 * v held 1e-6 mV away from 0 */
static inline double chr2_current(double v, const double* P, double O1, double O2) {
    double vs = v;
    if (fabs(v) < 1e-6) {
        if (v < 0.0) {
            vs = -1e-6;
        } else {
            vs = 1e-6;
        }
    }
    return P[RCSD_CHR2_GMAX] * (P[RCSD_CHR2_A] + P[RCSD_CHR2_B] * exp(v / P[RCSD_CHR2_C])) *
           (1.0 - P[RCSD_CHR2_E] / vs) * (O1 + P[RCSD_CHR2_GAMMA] * O2);
}

#endif
