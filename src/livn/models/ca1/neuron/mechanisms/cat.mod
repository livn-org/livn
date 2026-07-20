TITLE LVA (T-type) calcium (Ca2+) channel with high threshold for activation

COMMENT
Based on Magee and Johnston, 1995, J Physiol, 487:67-90, doi: 10.1113/jphysiol.1995.sp020862
Used in somatic and dendritic regions.
It calculates I_Ca using conductance (not permeability).

Important:
	The T-current does not activate calcium-dependent currents.
	The construction with dummy ion `Ca` prevents the updating of the internal calcium concentration.
ENDCOMMENT

NEURON {
	SUFFIX cat
	USEION ca READ cai, cao
	USEION Ca WRITE iCa VALENCE 2
	RANGE gcatbar, iCa
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
	(molar) = (1/liter)
	(mM) = (millimolar)
}

PARAMETER {:parameters that can be entered when function is called in hoc file
	tBase = 23.5  (degC)
	gcatbar = 0 (S/cm2) : initialized conductance
	ki = 0.001 (mM)
	tfa = 1 : activation time constant scaling factor
	tfi = 0.68 : inactivation time constant scaling factor
}

ASSIGNED {: parameters needed to solve DE
	v (mV)
	celsius (degC)	: initially was 22
	cai (mM)		: initial internal Ca++ concentration, initially 5.e-6
	cao (mM)		: initial external Ca++ concentration, initially 2
	eca (mV)		: Ca++ reversal potential, 140
	iCa (mA/cm2)
	gcat (S/cm2) 
	minf (1)
	hinf (1)
	taum (ms)
	tauh (ms)
}

STATE {	m h }: unknown activation and inactivation parameters to be solved in the DEs

BREAKPOINT {
	SOLVE states METHOD cnexp
	gcat = gcatbar*pow(m, 2)*h*h2(cai)	: maximum channel conductunce
	iCa = gcat*ghk(v, cai, cao)		: dummy calcium current induced by this channel
}

DERIVATIVE states { : exact when v held constant; integrates over dt step
	rates(v)
	m' = (minf - m)/taum
	h' = (hinf - h)/tauh
}

INITIAL {
	rates(v)
	m = minf
	h = hinf
}

FUNCTION h2(cai(mM)) {
	h2 = ki/(ki+cai)
}

FUNCTION ghk(v (mV), ci (mM), co (mM)) (mV) {
	LOCAL nu, f
	f = KTF(celsius)/2
	nu = v/f
	ghk = -f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (degC)) (mV) {: temperature-dependent adjustment factor
	KTF = (0.0853(mV/degC)*(celsius + 273.15(degC)))
}

FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
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

FUNCTION alph(v (mV)) (/ms) {
	alph = 1.6e-4(/ms)*exp(-(v+57(mV))/19(mV))
}

FUNCTION beth(v (mV)) (/ms) {
	beth = 1(/ms)/(exp(-(v-15(mV))/10(mV)) + 1.0)
}

FUNCTION alpm(v (mV)) (/ms) {
	alpm = 0.1967(/ms)*vtrap(-(v-19.88(mV)), 10.0(mV))
}

FUNCTION betm(v (mV)) (/ms) {
	betm = 0.046(/ms)*exp(-v/22.73(mV))
}

PROCEDURE rates(v (mV)) { :callable from hoc

	taum = 1/(tfa*(alpm(v) + betm(v)))	: estimation of activation tau
	minf = alpm(v)/(alpm(v)+betm(v))	: estimation of activation steady state

	tauh = 1/(tfi*(alph(v) + beth(v)))	: estimation of inactivation tau
	hinf = alph(v)/(alph(v)+beth(v))	: estimation of inactivation steady state
}
