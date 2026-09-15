TITLE AMPA and NMDA conductances of one terminal, driven by one NetCon

COMMENT
LinExp2Syn (AMPA) and LinExp2SynNMDA (NMDA) on the same segment, as one point
process. Both receptors of a connection see the same events at the same delay,
so a single NetCon can carry them: weight slots 0/1 are the AMPA weight and
unitary conductance, slots 2/3 the NMDA ones.

Each receptor keeps its own states, parameters and expressions exactly as in
its single-receptor mechanism, and both sum linearly across NetCons, so this is
the same model as the two point processes it replaces. It exists to halve the
NetCons a network needs and to let every connection of a segment share one
point process.
ENDCOMMENT

NEURON {
	POINT_PROCESS LinExp2SynAMPANMDA
	USEION ca READ eca WRITE ica
	RANGE tau_rise, tau_decay, e, g_ampa, i_ampa
	RANGE nmda_tau_rise, nmda_tau_decay, nmda_e, mg, vshift, Kd, gamma, pf
	RANGE g_nmda, i_nmda, pnmda
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(mM) = (milli/liter)
}

PARAMETER {
	: AMPA, as LinExp2Syn
	tau_rise = 1. (ms)
	tau_decay = 5. (ms)
	e = 0. (mV)
	: NMDA, as LinExp2SynNMDA
	nmda_tau_rise = 10. (ms) <1e-9,1e9>
	nmda_tau_decay = 35. (ms) <1e-9,1e9>
	nmda_e = 0 (mV)
	mg = 1 (mM)
	vshift = 0 (mV)
	Kd = 3.57 (mM)
	gamma = 0.062 (/mV)
	pf = 0.03 (1)
}

ASSIGNED {
	v (mV)
	i (nA)
	i_ampa (nA)
	i_nmda (nA)
	g_ampa (uS)
	g_nmda (uS)
	factor
	nmda_factor
	pnmda
	eca (mV)
	ica (nA)
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

	if (nmda_tau_rise/nmda_tau_decay > .9999) {
		nmda_tau_rise = .9999*nmda_tau_decay
	}
	C = 0
	D = 0
	tp = (nmda_tau_rise*nmda_tau_decay)/(nmda_tau_decay - nmda_tau_rise) * log(nmda_tau_decay/nmda_tau_rise)
	nmda_factor = -exp(-tp/nmda_tau_rise) + exp(-tp/nmda_tau_decay)
	nmda_factor = 1/nmda_factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g_ampa = B - A
	i_ampa = g_ampa * (v - e)
	g_nmda = D - C
	pnmda = mgblock(v)
	i_nmda = g_nmda*pnmda*(v - nmda_e)*(1-pf)
	ica = g_nmda*pnmda*(v - eca)*pf
	i = i_ampa + i_nmda
}

DERIVATIVE state {
	A' = -A/tau_rise
	B' = -B/tau_decay
	C' = -C/nmda_tau_rise
	D' = -D/nmda_tau_decay
}

FUNCTION mgblock(v(mV)) {
	mgblock = 1 / (1 + exp(gamma * -(v+vshift)) * (mg / Kd))
}

NET_RECEIVE(weight, g_unit (uS), nmda_weight, nmda_g_unit (uS)) {
	INITIAL {}
	A = A + weight*g_unit*factor
	B = B + weight*g_unit*factor
	C = C + nmda_weight*nmda_g_unit*nmda_factor
	D = D + nmda_weight*nmda_g_unit*nmda_factor
}
