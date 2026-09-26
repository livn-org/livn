TITLE Sodium Channel

COMMENT
Optional slow inactivation (s_on = 1): a slow gate s, g = gmax * minf^3 * h * s,
in the form Powers et al. 2012 use on a motoneuron (J Neurophysiol,
doi:10.1152/jn.00510.2011, after Fleidervish, Friedman & Gutnick 1996, J Physiol,
doi:10.1113/jphysiol.1996.sp021366):

    alpha_s = 0.001 exp(-(v+85)/30)            recovery
    beta_s  = 0.0034 / (1 + exp(-(v+17)/10))   entry, rising with depolarization
    s_inf   = s_floor + (1 - s_floor) alpha_s / (alpha_s + beta_s),  s_floor 0.4

Powers print beta_s as 0.034 exp(-(v+17)/10), which as written holds s at its
floor at every voltage. The sigmoid above is a reconstruction, checked against
Fleidervish's measured values: half-inactivation -42.6 mV (measured -43.8),
recovery tau 0.24-1.7 s over -128..-68 mV (0.45-2.5 s), onset 0.3-1.9 s when
depolarized (0.86-2.33 s).

s_speed scales both rates, leaving s_inf unchanged. At 1 this is the
neocortical time course. Motoneurons recover faster: 129.2 ms at -40 mV (Miles,
Dai & Brownstone 2005, J Physiol, doi:10.1113/jphysiol.2005.086033), which
s_speed ~14.5 matches. With s_on = 0 (default) the conductance is exactly
gmax * minf^3 * h.
ENDCOMMENT

NEURON {
	SUFFIX Nas
	USEION na READ ena WRITE ina
	RANGE gmax, ina, g, vhalf, slope
	RANGE s_on, s_floor, s_speed
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S)  = (siemens)
}

PARAMETER {
	gmax	=0.06 	(mho/cm2) <0,1e9>
	vhalf	=-35	(mV)
	slope	=7.8	(mV) <1e-9,1e9>
	s_on	=0
	s_floor	=0.4
	s_speed	=1	<1e-9,1e9>
}

ASSIGNED {
	v (mV)
	ena (mV)
	ina (mA/cm2)
	g (S/cm2)
	minf
	hinf htau
	sinf stau
}

STATE {
	h
	s
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	if (s_on > 0.5) {
		g = gmax * minf^3 * h * s
	} else {
		g = gmax * minf^3 * h
	}
	ina = g * (v - ena)
}

INITIAL { 
    rates(v)
    h = hinf
    s = sinf
}

DERIVATIVE states { 
	rates(v)
	h' = (hinf - h)/htau
	s' = (sinf - s)/stau
}

PROCEDURE rates(v(mV)) {LOCAL a, b, as, bs
        minf = 1/(1+exp(-(v-vhalf)/slope))
	htau = 30/(exp((v+50)/15)+exp(-(v+50)/16))
	hinf = 1/(1+exp((v+55)/7))
	as = s_speed * 0.001 * exp(-(v+85)/30)
	bs = s_speed * 0.0034 / (1 + exp(-(v+17)/10))
	stau = 1/(as + bs)
	sinf = s_floor + (1 - s_floor) * as/(as + bs)
}