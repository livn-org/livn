TITLE HVA (L-type) calcium channel with low threshold for activation (soma)

COMMENT
Used in somatic and proximal dendritic regions.
Uses channel conductance (not permeability).
ENDCOMMENT

NEURON {
	SUFFIX cal
	USEION ca READ cai, cao WRITE ica
	RANGE gcalbar, ica
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
	(molar) = (1/liter)
	(mM) = (millimolar)
}

PARAMETER {	:parameters that can be entered when function is called in cell-setup
	gcalbar = 0 (S/cm2)	: initialized conductance
	ki = 0.001 (mM)
	tfa = 5				: time constant scaling factor
}

ASSIGNED { : parameters needed to solve DE
	v (mV)
	eca (mV)		: Ca2+ reversal potential
	cai (mM)		: initial internal Ca2+ concentration
	cao (mM)		: initial external Ca2+ concentration
	celsius (degC)	: temperature

	ica (mA/cm2)
	gcal (S/cm2)
	minf (1)
	taum (ms)
}

STATE {	: unknown parameter to be solved in the DEs
	m
} 

BREAKPOINT {
	SOLVE states METHOD cnexp
	gcal = gcalbar*m*h2(cai)	: maximum channel permeability
	ica = gcal*ghk(v, cai, cao)	: calcium current induced by this channel
}

DERIVATIVE states {
	rates (v)
	m' = (minf - m)/taum
}

INITIAL {: initialize the following parameter using rates()
	rates(v)
	m = minf
}

FUNCTION h2(cai (mM)) {
	h2 = ki/(ki+cai)
}

FUNCTION ghk(v (mV), ci (mM), co (mM)) (mV) {
	LOCAL nu, f
	f = KTF(celsius)/2
	nu = v/f
	ghk = -f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (degC)) (mV) {	: temperature-dependent adjustment factor
	KTF = (0.0853(mV/degC)*(celsius + 273.15(degC)))
}

FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	} else {
		efun = z/(exp(z) - 1)
	}
}

FUNCTION vtrap(x (mV), y (mV)) (1) {
	:Traps for 0 in denominator of rate eqns. Taylor expansion is used.
	if (fabs(x/y) < 1e-6) {
		vtrap = 1(/mV)*y*(1 - x/y/2)
	} else {  
		vtrap = 1(/mV)*x/(exp(x/y) - 1)
	}
}

FUNCTION alpm(v (mV)) (/ms){
	alpm = 0.055(/ms)*vtrap(-(v+27.01(mV)), 3.8(mV))
}

FUNCTION betm(v (mV)) (/ms){
	betm =0.94(/ms)*exp(-(v + 63.01(mV))/17(mV))
}

PROCEDURE rates(v (mV)) { :callable from hoc
	taum = 1/(tfa*(alpm(v)+betm(v)))	: estimation of activation tau
	minf = alpm(v)/(alpm(v)+betm(v))	: estimation of activation steady state value
}
