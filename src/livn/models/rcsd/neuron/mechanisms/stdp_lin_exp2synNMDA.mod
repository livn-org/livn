TITLE LinExp2SynNMDA with spike-timing dependent plasticity (STDP)

COMMENT

Extends LinExp2SynNMDA with a voltage-based STDP rule (see StdpLinExp2Syn).

The plasticity rule is adapted from the ExcSigma3Exp2Syn mechanism from neuronpp
(https://github.com/ziemowit-s/neuronpp).

Identical NMDA conductance model (dual-exponential with Mg2+ block)
plus a Hebbian learning rule that modifies each connection's weight
`w_plastic` based on postsynaptic voltage.

Each incoming NetCon connection carries its own plastic weight
`w_plastic` (3rd weight vector element) and a per-connection
integral snapshot `last_int` (4th element). Weight updates happen at spike arrival:

  delta = learn_int - last_int
  w_plastic = w_plastic + delta * w_plastic

This captures the total learning since the previous spike on this specific connection,
enabling per-connection differentiation.

ENDCOMMENT

NEURON {
    POINT_PROCESS StdpLinExp2SynNMDA
    RANGE vshift, Kd, gamma, mg
    RANGE tau_rise, tau_decay, e, i
    RANGE g, pnmda
    RANGE w, w_init
    RANGE learning_w, ltd, ltp, learn_int
    RANGE A_ltp, A_ltd
    RANGE theta_ltp, theta_ltd
    RANGE ltp_sigmoid_half, ltd_sigmoid_half
    RANGE learning_slope, learning_tau
    RANGE w_max, w_min
    RANGE plasticity_on
    NONSPECIFIC_CURRENT i
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
    (mM) = (milli/liter)
}

PARAMETER {
    : -- Synaptic kinetics (same as LinExp2SynNMDA) --
    tau_rise    = 10.   (ms) <1e-9,1e9>
    tau_decay   = 35.   (ms) <1e-9,1e9>
    e           = 0     (mV)
    mg          = 1     (mM)  : external magnesium concentration
    vshift      = 0     (mV)  : positive left-shifts mg unblock
    Kd          = 3.57  (mM)  : Mg concentration dependence
    gamma       = 0.062 (/mV) : slope of Mg sensitivity

    : -- Plasticity parameters --
    plasticity_on = 0         : 0 = off, 1 = on

    w_init      = 1.0         : initial weight multiplier
    A_ltp       = 1.0         : LTP amplitude scaling
    A_ltd       = 1.0         : LTD amplitude scaling
    theta_ltp   = -45.  (mV)  : voltage threshold for LTP
    theta_ltd   = -60.  (mV)  : voltage threshold for LTD
    ltp_sigmoid_half = -40. (mV) : sigmoid half-activation for LTP
    ltd_sigmoid_half = -55. (mV) : sigmoid half-activation for LTD
    learning_slope = 1.3      : sigmoid slope
    learning_tau   = 20.      : learning signal time scale
    w_max       = 5.0         : maximum weight
    w_min       = 0.0001      : minimum weight
}

ASSIGNED {
    v       (mV)
    i       (nA)
    g       (uS)
    factor
    pnmda
    ltd
    ltp
    w : diagnostic: last-updated per-connection w_plastic
}

STATE {
    A (uS)
    B (uS)
    learning_w
    learn_int : running integral of learning_w
}

INITIAL {
    LOCAL tp
    if (tau_rise/tau_decay > .9999) {
        tau_rise = .9999 * tau_decay
    }
    A = 0
    B = 0
    tp = (tau_rise * tau_decay) / (tau_decay - tau_rise) * log(tau_decay / tau_rise)
    factor = -exp(-tp / tau_rise) + exp(-tp / tau_decay)
    factor = 1 / factor

    ltd = 0
    ltp = 0
    learning_w = 0
    learn_int = 0
    w = w_init
}

BREAKPOINT {
    SOLVE state METHOD cnexp

    g = B - A
    pnmda = mgblock(v)
    i = g * pnmda * (v - e)

    if (plasticity_on > 0) {
        : Compute LTD/LTP signals from voltage (shared across connections)
        if (v - theta_ltd > 0) {
            ltd = sigmoid_thr(learning_slope, v, ltd_sigmoid_half)
        } else {
            ltd = 0
        }
        if (v - theta_ltp > 0) {
            ltp = sigmoid_thr(learning_slope, v, ltp_sigmoid_half)
        } else {
            ltp = 0
        }

        : Accumulate shared learning signal
        learning_w = learning_w + sigmoid_sat(learning_slope, (-A_ltd * ltd + A_ltp * 2 * ltp) / learning_tau) / 5000
    }
}

DERIVATIVE state {
    A' = -A / tau_rise
    B' = -B / tau_decay
    learning_w' = -learning_w / 4
    learn_int' = learning_w
}

NET_RECEIVE(weight, g_unit (uS), w_plastic, last_int) {
    INITIAL {
        w_plastic = w_init
        last_int = 0
    }
    : Presynaptic spike - apply per-connection learning and update conductance
    if (plasticity_on > 0.5) {
        LOCAL delta_learn
        delta_learn = learn_int - last_int
        w_plastic = w_plastic + delta_learn * w_plastic
        last_int = learn_int
        if (w_plastic > w_max) {
            w_plastic = w_max
        }
        if (w_plastic < w_min) {
            w_plastic = w_min
        }
        w = w_plastic
    }
    A = A + w_plastic * weight * g_unit * factor
    B = B + w_plastic * weight * g_unit * factor
}

FUNCTION mgblock(v(mV)) {
    : from Jahr & Stevens
    mgblock = 1 / (1 + exp(gamma * -(v + vshift)) * (mg / Kd))
}

: sigmoid with threshold
FUNCTION sigmoid_thr(slope, value, thr) {
    sigmoid_thr = 1 / (1.0 + pow(slope, -(value - thr)))
}

: sigmoidal saturation [-1, 1]
FUNCTION sigmoid_sat(slope, value) {
    sigmoid_sat = 2.0 / (1.0 + pow(slope, -value)) - 1.0
}
