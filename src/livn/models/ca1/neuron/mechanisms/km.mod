TITLE Slowly activating potassium current (muscarinic K+ channel)

COMMENT
Potassium channel, Hodgkin-Huxley style kinetics.
Voltage-dependent potassium current (Im) (muscarinic K+ channel)
Slow, noninactivating.

Author: Zach Mainen, Salk Institute, 1995, zach@salk.edu
ENDCOMMENT

NEURON {
	SUFFIX km
	USEION k READ ek WRITE ik
	RANGE gbar, ik
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
}

PARAMETER {
	gbar = 0.03 (S/cm2)
	tha  = -30 (mV)			: v1/2 for ninf
	qa   = 9 (mV)			: ninf slope
	Ra   = 0.001 (/ms/mV)	: max act rate  (slow)
	Rb   = 0.001 (/ms/mV)	: max deact rate  (slow)
	temp = 23	(degC)		: original temp
	q10  = 2.3				: temperature sensitivity
}

ASSIGNED {
	v (mV)
	gk (S/cm2)
	ek (mV)
	celsius (degC)
	ik (mA/cm2)
	tadj
	ninf
	taun (ms)
}

STATE {
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	tadj = q10^((celsius - temp)/10(degC))  :temperature adjastment
	gk = tadj*gbar*n
	ik = gk * (v - ek)
}

DERIVATIVE states { : Computes state variable n at the current v and dt.
	rates(v)
	n' = (ninf - n)/taun
}

INITIAL { 
	rates(v)
	n = 0
}

FUNCTION vtrap(x (mV), y (mV)) (1) {
	:Traps for 0 in denominator of rate eqns. Taylor expansion is used.
	if (fabs(x/y) < 1e-6) {
		vtrap = 1(/mV)*y*(1 - x/y/2)
	} else {  
		vtrap = 1(/mV)*x/(exp(x/y) - 1)
	}
}

FUNCTION alpha(v (mV)) (/ms) { :callable from hoc
	alpha = Ra * vtrap(-(v - tha), qa)
}

FUNCTION beta(v (mV)) (/ms) { :callable from hoc
	beta = Rb * vtrap(v - tha, qa)
}

PROCEDURE rates(v (mV)) {
	:Computes rate and other constants at current v.
	:Call once from HOC to initialize inf at resting v.
	taun = 1/(alpha(v) + beta(v))
	ninf = alpha(v)*taun
}