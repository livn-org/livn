TITLE Two linear double-exponential conductances of one terminal, driven by one NetCon

COMMENT
Two LinExp2Syn on the same segment, as one point process -- GABA_A and GABA_B
of one connection. Both see the same events at the same delay, so a single
NetCon carries them: weight slots 0/1 are the first receptor's weight and
unitary conductance, slots 2/3 the second's.

Each receptor keeps its own states, parameters and expressions exactly as in
LinExp2Syn, and both sum linearly across NetCons, so this is the same model as
the two point processes it replaces.
ENDCOMMENT

NEURON {
	POINT_PROCESS LinExp2SynPair
	RANGE tau_rise, tau_decay, e, g1, i1
	RANGE tau_rise2, tau_decay2, e2, g2, i2
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau_rise = 1. (ms)
	tau_decay = 5. (ms)
	e = 0. (mV)
	tau_rise2 = 1. (ms)
	tau_decay2 = 5. (ms)
	e2 = 0. (mV)
}

ASSIGNED {
	v (mV)
	i (nA)
	i1 (nA)
	i2 (nA)
	g1 (uS)
	g2 (uS)
	factor
	factor2
}

STATE {
	A (uS)
	B (uS)
	C (uS)
	D (uS)
}

INITIAL {
	LOCAL tp
	if (tau_rise/tau_decay > 0.9999) {
		tau_rise = 0.9999*tau_decay
	}
	if (tau_rise/tau_decay < 1e-9) {
		tau_rise = tau_decay*1e-9
	}
	A = 0
	B = 0
	tp = (tau_rise*tau_decay)/(tau_decay - tau_rise) * log(tau_decay/tau_rise)
	factor = -exp(-tp/tau_rise) + exp(-tp/tau_decay)
	factor = 1/factor

	if (tau_rise2/tau_decay2 > 0.9999) {
		tau_rise2 = 0.9999*tau_decay2
	}
	if (tau_rise2/tau_decay2 < 1e-9) {
		tau_rise2 = tau_decay2*1e-9
	}
	C = 0
	D = 0
	tp = (tau_rise2*tau_decay2)/(tau_decay2 - tau_rise2) * log(tau_decay2/tau_rise2)
	factor2 = -exp(-tp/tau_rise2) + exp(-tp/tau_decay2)
	factor2 = 1/factor2
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g1 = B - A
	i1 = g1 * (v - e)
	g2 = D - C
	i2 = g2 * (v - e2)
	i = i1 + i2
}

DERIVATIVE state {
	A' = -A/tau_rise
	B' = -B/tau_decay
	C' = -C/tau_rise2
	D' = -D/tau_decay2
}

NET_RECEIVE(weight, g_unit (uS), weight2, g_unit2 (uS)) {
	INITIAL {}
	A = A + weight*g_unit*factor
	B = B + weight*g_unit*factor
	C = C + weight2*g_unit2*factor2
	D = D + weight2*g_unit2*factor2
}
