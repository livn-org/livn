TITLE Shared synaptic conductance with per-stream saturation

COMMENT

Milstein, 2018

Rise and decay kinetics are shared across all presynaptic sources. Conductances are linearly summed across sources, and
updated at every time step. Each source stores an independent weight multiplier, unitary peak conductance (g_unit), and
memory of its contribution to the total shared conductance. These per-stream parameters need only be updated when a
spike arrives from that source. This allows each stream to independently saturate its conductance during repetitive
activation.

Implementation informed by:

The NEURON Book: Chapter 10, N.T. Carnevale and M.L. Hines, 2004

ENDCOMMENT

NEURON {
	POINT_PROCESS SatExp2Syn
	RANGE g, i, dur_onset, tau_offset, e, sat, g_inf
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
	(mM) = (milli/liter)
}

PARAMETER {
	sat 			= 0.9 			: target saturation at peak of single event
	dur_onset		= 1.	(ms) 	: time to peak of single event, determines tau_onset
	tau_offset 		= 5. 	(ms) 	: time constant of exponential decay
	e 				= 0. 	(mV) 	: reversal potential
}


ASSIGNED {
	v			(mV)		: postsynaptic voltage
	i 			(nA)		: current = g*(v - Erev)
	g 			(umho)		: conductance
    g_inf 					: steady-state fraction active channels
	tau_onset 	(ms)		: time constant of exponential rise
    alpha 		(/ms) 		: kinetic rate of rise
	beta 		(/ms) 		: kinetic rate of decay
 	syn_onset 	(umho)		: total weight of synapses in onset phase
}

STATE {
	g_onset 	(umho) 		: conductance of synapses in onset phase
	g_offset 	(umho) 		: conductance of synapses in offset phase
}

INITIAL {
	beta = 1. / tau_offset
	tau_onset =  -dur_onset / log(1. - sat)
	alpha = 1. / tau_onset - beta
 	g_inf = alpha / (alpha + beta)
	syn_onset = 0.
        g_onset = 0
        g_offset = 0
}

BREAKPOINT {
	SOLVE release METHOD cnexp
	g = (g_onset + g_offset) / sat / g_inf
	i = g * (v - e)
}

DERIVATIVE release {
	g_onset' = (syn_onset * g_inf - g_onset) / tau_onset
	g_offset' = -g_offset / tau_offset
}

: following supports both saturation from single input and
: summation from multiple inputs
: if spike occurs during onset_dur then new peak time is t + onset_dur
: onset phase concatenates but does not summate

NET_RECEIVE(weight, g_unit (umho), onset, count, g0, t0 (ms)) {
	INITIAL {
		onset = 0
		count = 0
		g0 = 0.
		t0 = 0.
	}
	: flag is an implicit argument of NET_RECEIVE and normally 0
	if (flag == 0) { : a spike, begin onset phase if not already in onset phase
		count = count + 1
		if (!onset) {
			g0 = g0*exp(-(t - t0)/tau_offset)
			t0 = t
			onset = 1
			syn_onset = syn_onset + weight * g_unit
			g_onset = g_onset + g0 * weight * g_unit
			g_offset = g_offset - g0 * weight * g_unit
		}
	: come again in dur_onset with flag = current count
	net_send(dur_onset, count)
	}
	else {
		if (flag == count) { : if the offset signal is associated with the most recent spike then begin offset phase
			g0 = g_inf - (g_inf - g0) * exp(-(t - t0)/tau_onset)
			t0 = t
			syn_onset = syn_onset - weight * g_unit
			g_onset = g_onset - g0 * weight * g_unit
			g_offset = g_offset + g0 * weight * g_unit
			onset = 0
		}
	}
}
