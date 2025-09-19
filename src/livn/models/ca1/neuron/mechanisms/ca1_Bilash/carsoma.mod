TITLE HVAm (R-type) calcium (CA2+) channel with medium threshold for activation

COMMENT
Used in somatic regions.
It has lower threshold for activation/inactivation and slower activation time constant,
than the same mechanism in dendritic regions.
Uses channel conductance (not permeability).
Written by Yiota Poirazi on 3/12/01 poirazi@LNC.usc.edu.
ENDCOMMENT

NEURON {
	SUFFIX carsoma
	USEION ca READ eca WRITE ica
	RANGE gcabar, ica
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
}

PARAMETER { : parameters that can be entered when function is called in cell-setup
	gcabar = 0 (S/cm2) : initialized conductance
}

STATE { : unknown activation and inactivation parameters to be solved in the DEs
	m
	h
}

ASSIGNED { : parameters needed to solve DE
	v (mV)
	eca (mV)	: Ca2+ reversal potential
	ica (mA/cm2)
	minf (1)
	hinf (1)
	taum (ms)
	tauh (ms)
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ica = gcabar*pow(m, 3)*h*(v - eca)
}

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/taum
	h' = (hinf - h)/tauh
}

INITIAL {
	rates(v)
	m = 0	: initial activation parameter value
	h = 1	: initial inactivation parameter value
}

PROCEDURE rates(v (mV)) {
	minf = 1 / (1 + exp((v + 55(mV))/(-3(mV))))	:Ca activation
	hinf = 1 / (1 + exp((v + 57(mV))/(1(mV))))	:Ca inactivation
	taum = 100.0
	tauh = 5.0
}
