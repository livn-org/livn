TITLE A-type potassium channel for V1 spinal interneurons

NEURON {
    SUFFIX Ka_v1in
    USEION k READ ek WRITE ik
    RANGE gmax, ik, g
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S)  = (siemens)
}

PARAMETER {
    gmax = 0.0 (mho/cm2) <0,1e9>
}

ASSIGNED {
    v    (mV)
    ek   (mV)
    ik   (mA/cm2)
    g    (S/cm2)
    ainf
    atau (ms)
    binf
    btau (ms)
}

STATE { a b }

BREAKPOINT {
    SOLVE states METHOD cnexp
    g  = gmax * a * b
    ik = g * (v - ek)
}

INITIAL {
    rates(v)
    a = ainf
    b = binf
}

DERIVATIVE states {
    rates(v)
    a' = (ainf - a) / atau
    b' = (binf - b) / btau
}

PROCEDURE rates(v(mV)) {
    : Activation: half at -36 mV, fast (1 ms) -  onset at sub-threshold voltages
    : Inactivation: half at -66 mV, slow (15 ms) -  drives inter-spike adaptation
    ainf = 1 / (1 + exp(-(v + 36) / 8))
    atau = 1.0
    binf = 1 / (1 + exp( (v + 66) / 8))
    btau = 15.0
}
