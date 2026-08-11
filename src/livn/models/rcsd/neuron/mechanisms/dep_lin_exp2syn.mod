TITLE Shared synaptic conductance with per-stream linear summation and short-term depression

COMMENT

`LinExp2Syn` (Milstein, 2018) with Tsodyks-Markram depression added per
presynaptic stream.

Rise and decay kinetics are shared across all presynaptic sources, as in
`LinExp2Syn`. What each stream now carries in addition is its own pool of
available resources `R`, depleted by a fraction `U` on every event and
recovering exponentially with `tau_rec` between them:

    R  <-  1 - (1 - R) * exp(-(t - tlast) / tau_rec)     recover
    g  <-  g + weight * g_unit * R * U * factor          release
    R  <-  R * (1 - U)                                   deplete

Recovery is evaluated at event arrival rather than integrated as a STATE, so
depression costs no per-timestep work and no state variable per connection.

With `U = 1` and `tau_rec` short this reduces to `LinExp2Syn` scaled by
`weight * g_unit`, so the depressing and non-depressing forms stay comparable.

Implementation informed by:

Tsodyks & Markram, PNAS 1997
The NEURON Book: Chapter 10, N.T. Carnevale and M.L. Hines, 2004

ENDCOMMENT

NEURON {
	POINT_PROCESS DepLinExp2Syn
	RANGE g, i, tau_rise, tau_decay, e
	RANGE U, tau_rec
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
	(mM) = (milli/liter)
}

PARAMETER {
	tau_rise		= 1.	(ms) 	: time constant of exponential rise
	tau_decay 		= 5. 	(ms) 	: time constant of exponential decay
	e 				= 0. 	(mV) 	: reversal potential
	U				= 0.25			: fraction of available resources released per event
	tau_rec			= 400.	(ms)	: recovery time constant of the resource pool
}


ASSIGNED {
	v			(mV)		: postsynaptic voltage
	i 			(nA)		: current = g*(v - Erev)
	g 			(umho)		: conductance
    factor 					: normalization factor

}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
	LOCAL tp
	if (tau_rise/tau_decay > 0.9999) {
		tau_rise = 0.9999*tau_decay
	}
	if (tau_rise/tau_decay < 1e-9) {
		tau_rise = tau_decay*1e-9
	}
	if (U <= 0) {
		U = 1e-6
	}
	if (U > 1) {
		U = 1
	}
	if (tau_rec <= 0) {
		tau_rec = 1e-3
	}
	A = 0
	B = 0
	tp = (tau_rise*tau_decay)/(tau_decay - tau_rise) * log(tau_decay/tau_rise)
	factor = -exp(-tp/tau_rise) + exp(-tp/tau_decay)
	factor = 1/factor
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	g = B - A
	i = g * (v - e)
}

DERIVATIVE state {
	A' = -A/tau_rise
	B' = -B/tau_decay
}

: `R` and `tlast` are per-stream state held in the NetCon weight vector, so
: every presynaptic source depresses and recovers independently even though
: they share this point process and its conductance.
NET_RECEIVE(weight, g_unit (umho), R, tlast (ms)) {
	INITIAL {
		R = 1
		tlast = -1e9
	}
	R = 1 - (1 - R) * exp(-(t - tlast) / tau_rec)
	tlast = t
	A = A + weight * g_unit * R * U * factor
	B = B + weight * g_unit * R * U * factor
	R = R - R * U
}
