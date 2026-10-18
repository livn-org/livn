TITLE Channelrhodopsin-2 (H134R), four-state model

COMMENT
Williams JC, Xu J, Lu Z, Klimas A, Chen X, Ambrosi CM, Cohen IS, Entcheva E
(2013) Computational optogenetics: empirically-derived voltage- and
light-sensitive channelrhodopsin-2 model. PLoS Comput Biol 9(9): e1003220.

Ported from the NEURON implementation by M. Giugliano (ModelDB 244414). The
states, rates, Q10 scaling and cnexp scheme are the original's; it differs in:

- Irradiance (mW/mm2) is set by livn every step; the original's built-in
  single pulse (light_delay, pulse_width, light_intensity) is gone.
- gmax is in S/cm2, livn's conductance-density unit: 4e-4 is the original's
  0.4 mS/cm2, and the 1e-3 in the current goes with it.
- The rectification (A + B exp(v/C))/v * (v - EChR2) is evaluated as
  (A + B exp(v/C)) * (1 - EChR2/v), the same function without the 0/0 at
  v = 0; within 1e-6 mV of 0 it takes v = +-1e-6, which is exact for
  EChR2 = 0 and keeps the current finite otherwise.
- Every parameter is RANGE, so that each cell can carry its own (gmax is the
  per-cell expression), and the two whose units are not livn's defaults carry
  them: sigma_retinal_m2 and hc_J_m.

The rates scale with Q10 from temp (22 degC) to celsius, which livn sets from
the model before finitialize.
ENDCOMMENT

NEURON {
    SUFFIX ChR2H134R
    NONSPECIFIC_CURRENT i
    RANGE i, gmax, Irradiance, EChR2, gamma, A, B, C
    RANGE wavelength, hc_J_m, wloss, sigma_retinal_m2, tauChR2, temp
    RANGE Q10_Gd1, Q10_Gd2, Q10_Gr, Q10_e12dark, Q10_e21dark, Q10_epsilon1, Q10_epsilon2
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gmax = 4e-4         (S/cm2)
    EChR2 = 0           (mV)
    gamma = 0.1                     : O2/O1 conductance ratio

    A = 10.6408         (mV)
    B = -14.6408        (mV)
    C = -42.7671        (mV)

    wavelength = 470                : nm
    hc_J_m = 1.986446e-25           : Planck's constant times c, J m
    wloss = 1.3                     : photon losses to scattering and absorption
    sigma_retinal_m2 = 12e-20       : retinal cross-section, m2

    tauChR2 = 1.3       (ms)

    temp = 22           (degC)
    Q10_Gd1 = 1.97
    Q10_Gd2 = 1.77
    Q10_Gr = 2.56
    Q10_e12dark = 1.1
    Q10_e21dark = 1.95
    Q10_epsilon1 = 1.46
    Q10_epsilon2 = 2.77
}

ASSIGNED {
    Irradiance                      : mW/mm2, set by livn
    v           (mV)
    celsius     (degC)
    i           (mA/cm2)
    Gd1         (/ms)
    Gd2         (/ms)
    Gr          (/ms)
    e12         (/ms)
    e21         (/ms)
    epsilon1
    epsilon2
    F           (/ms)
    S0
}

STATE {
    O1 O2 C1 p
}

BREAKPOINT {
    LOCAL vs
    SOLVE states METHOD cnexp
    vs = v
    if (fabs(v) < 1e-6) {
        if (v < 0) {
            vs = -1e-6
        } else {
            vs = 1e-6
        }
    }
    i = gmax * (A + B * exp(v / C)) * (1 - EChR2 / vs) * (O1 + gamma * O2)
}

INITIAL {
    Irradiance = 0
    rates(v)
    C1 = 1
    O1 = 0
    O2 = 0
    p = 0
}

DERIVATIVE states {
    rates(v)
    : C2 = 1 - O1 - O2 - C1
    O1' = -(Gd1 + e12) * O1 + e21 * O2 + epsilon1 * F * p * C1
    O2' = -(Gd2 + e21) * O2 + e12 * O1 + epsilon2 * F * p * (1 - C1 - O1 - O2)
    C1' = -epsilon1 * F * p * C1 + Gd1 * O1 + Gr * (1 - C1 - O1 - O2)
    p' = (S0 - p) / tauChR2
}

UNITSOFF
PROCEDURE rates(v (mV)) {
    LOCAL e12dark, e21dark, logphi0, Ephoton, flux

    : at 22 degC
    e12dark = 0.011
    e21dark = 0.008
    epsilon1 = 0.8535
    epsilon2 = 0.14
    Gd1 = 0.075 + 0.043 * tanh(-(v + 20) / 20)
    Gd2 = 0.05
    Gr = 0.0000434587 * exp(-0.0211539274 * v)

    e12dark = e12dark * Q10_e12dark^((celsius - temp) / 10)
    e21dark = e21dark * Q10_e21dark^((celsius - temp) / 10)
    epsilon1 = epsilon1 * Q10_epsilon1^((celsius - temp) / 10)
    epsilon2 = epsilon2 * Q10_epsilon2^((celsius - temp) / 10)
    Gd1 = Gd1 * Q10_Gd1^((celsius - temp) / 10)
    Gd2 = Gd2 * Q10_Gd2^((celsius - temp) / 10)
    Gr = Gr * Q10_Gr^((celsius - temp) / 10)

    if (Irradiance > 0) {
        logphi0 = log(1 + Irradiance / 0.024)
    } else {
        logphi0 = 0
    }
    e12 = e12dark + 0.005 * logphi0
    e21 = e21dark + 0.004 * logphi0

    S0 = 0.5 * (1 + tanh(120 * (100 * Irradiance - 0.1)))
    Ephoton = 1e9 * hc_J_m / wavelength
    flux = 1000 * Irradiance / Ephoton
    F = flux * sigma_retinal_m2 / (wloss * 1000)
}
UNITSON
