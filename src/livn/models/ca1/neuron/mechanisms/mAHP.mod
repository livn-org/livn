TITLE Slow calcium-dependent potassium current (mAHP)

COMMENT
Ca2+-dependent K+ channel medium AHP.
From Moczydlowski and Latorre, 1983, J Gen Physiol, 82: 511-542, doi: 10.1085/jgp.82.4.511
ENDCOMMENT

NEURON {
	SUFFIX mAHP
	USEION ca READ cai
	USEION k READ ek WRITE ik
	RANGE gkbar, ik
}

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
	(molar) = (1/liter)
	(mM) = (millimolar)
	FARADAY = (faraday) (kilocoulombs)
	R = (k-mole) (joule/degC)
}

PARAMETER {
	gkbar = 0.01 (S/cm2)
	d1 = 0.84 (1)
	d2 = 1.0 (1)
	k1 = 0.18 (mM)
	k2 = 0.011 (mM)
	bbar = 0.28 (/ms)
	abar = 0.48 (/ms)
}

ASSIGNED {
	cai (mM) : typically 0.001
	celsius (degC) : typically 20
	v (mV)
	ek (mV)
	ik (mA/cm2)
	oinf
	tau (ms)
}

STATE {	: fraction of open channels
	o
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	ik = gkbar*o*(v - ek) : potassium current induced by this channel
}

DERIVATIVE state {
	rate(v, cai)
	o' = (oinf - o)/tau
}

INITIAL {
	rate(v, cai)
	o = oinf
}

FUNCTION alp(v (mV), ca (mM)) (1/ms) { :callable from hoc
	alp = abar/(1 + exp1(k1, d1, v)/ca)
}

FUNCTION bet(v (mV), ca (mM)) (1/ms) { :callable from hoc
	bet = bbar/(1 + ca/exp1(k2, d2, v))
}  

FUNCTION exp1(k (mM), d (1), v (mV)) (mM) { :callable from hoc
	exp1 = k*exp(-2*d*v*FARADAY/(R*(273.15(degC) + celsius)))
}

PROCEDURE rate(v (mV), ca (mM)) { :callable from hoc
	LOCAL a
	a = alp(v, ca)
	tau = 1/(a + bet(v, ca))	: estimation of activation tau
	oinf = a*tau				: estimation of activation steady state value
}
