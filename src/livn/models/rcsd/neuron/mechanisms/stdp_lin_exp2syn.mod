TITLE LinExp2Syn with spike-timing dependent plasticity (STDP)

COMMENT

Extends LinExp2Syn (Milstein, 2018) with a voltage-based STDP rule.

The plasticity rule is adapted from the ExcSigma3Exp2Syn mechanism from neuronpp
(https://github.com/ziemowit-s/neuronpp)

Each incoming NetCon connection carries its own plastic weight `w_plastic` 
(3rd element of the NetCon weight vector) and a per-connection integral 
snapshot `last_int` (4th element). This enables per-connection Hebbian 
learning even when multiple presynaptic sources target the same point process.

When plasticity_on == 1 the learning signal `learning_w` is accumulated 
in BREAKPOINT from postsynaptic voltage via sigmoidal LTD/LTP detectors. 
A running integral of learning_w (`learn_int`) is maintained as a STATE variable.

Weight update happens at spike arrival in NET_RECEIVE:

  delta = learn_int - last_int
  w_plastic = w_plastic + delta * w_plastic

This captures the total learning that occurred since the previous
spike on this specific connection. Because different presynaptic
neurons fire at different times (and thus sample different portions
of the learning-signal waveform), weights differentiate naturally.

When plasticity_on == 0 the synapse behaves identically to LinExp2Syn.

ENDCOMMENT

NEURON {
    POINT_PROCESS StdpLinExp2Syn
    RANGE g, i, tau_rise, tau_decay, e
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
    (umho) = (micromho)
}

PARAMETER {
    : -- Synaptic kinetics (same as LinExp2Syn) --
    tau_rise    = 1.    (ms)  : rise time constant
    tau_decay   = 5.    (ms)  : decay time constant
    e           = 0.    (mV)  : reversal potential

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
    g       (umho)
    factor
    ltd
    ltp
    w             : diagnostic: last-updated per-connection w_plastic
}

STATE {
    A (uS)
    B (uS)
    learning_w
    learn_int     : running integral of learning_w
}

INITIAL {
    LOCAL tp
    if (tau_rise/tau_decay > 0.9999) {
        tau_rise = 0.9999 * tau_decay
    }
    if (tau_rise/tau_decay < 1e-9) {
        tau_rise = tau_decay * 1e-9
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
    i = g * (v - e)

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

NET_RECEIVE(weight, g_unit (umho), w_plastic, last_int) {
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

: sigmoid with threshold
FUNCTION sigmoid_thr(slope, value, thr) {
    sigmoid_thr = 1 / (1.0 + pow(slope, -(value - thr)))
}

: sigmoidal saturation [-1, 1]
FUNCTION sigmoid_sat(slope, value) {
    sigmoid_sat = 2.0 / (1.0 + pow(slope, -value)) - 1.0
}
