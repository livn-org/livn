TITLE Slow calcium-dependent potassium current (sAHP)

COMMENT
Ca2+-dependent K+ current responsible for slow AHP

Sah and Bekkers, 1996, J Neurosci 16:4537-4542, doi: 10.1523/JNEUROSCI.16-15-04537.1996

Differential equations

Model based on a first order kinetic scheme

		+n cai <->     (alpha, beta)

Following this model, the activation function will be half-activated at
a concentration of Cai = (beta/alpha)^(1/n) = cac (parameter)

The mod file here is written for the case n=2 (2 binding sites)
---------------------------------------------

This current models the "slow" IK[Ca] (IAHP): 
	- potassium current
	- activated by intracellular calcium
	- NOT voltage dependent

A minimal value for the time constant has been added

Ref: Destexhe et al., 1994, J Neurophysiol 72: 803-818, doi: 10.1152/jn.1994.72.2.803
See also: http://www.cnl.salk.edu/~alain , http://cns.fmed.ulaval.ca

Modifications by Yiota Poirazi 2001 (poirazi@LNC.usc.edu)
	taumin = 0.5 ms instead of 0.1 ms	
ENDCOMMENT

NEURON {
	SUFFIX kca
	USEION k READ ek WRITE ik
	USEION ca READ cai
	RANGE gbar, ik
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
	(molar) = (1/liter)
	(mM) = (millimolar)
}

PARAMETER {
	gbar = 0.01 (S/cm2)
	beta = 0.03 (1/ms)	: backward rate constant
	cac = 0.025 (mM)	: middle point of activation function
	taumin = 0.5 (ms)	: minimal value of the time constant
	q10 = 3 (1) 		: temperature adjacent constant
}

ASSIGNED {: parameters needed to solve DE
	v (mV)
	celsius (degC)
	ek (mV)
	cai (mM)	: initial [Ca]i
	gk (S/cm2)
	ik (mA/cm2)
	minf
	taum (ms)
}

STATE {: activation variable to be solved in the DEs
	m
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	gk = gbar*pow(m, 3)		: maximum channel conductance
	ik = gk*(v - ek)	: potassium current induced by this channel
}

DERIVATIVE states { 
	rates(cai)
	m' = (minf - m) / taum
}

INITIAL {
	rates(cai)
	m = minf
}

PROCEDURE rates(cai (mM)) {
	LOCAL car, tadj
	tadj = q10^((celsius-22.0(degC))/10(degC))		: temperature-dependent adjastment factor
	car = (cai/cac)^2
	minf = car / ( 1 + car )			: activation steady state value
	taum =  1 / beta / (1 + car) / tadj	: activation min value of time constant
	if (taum < taumin) {taum = taumin}
}
