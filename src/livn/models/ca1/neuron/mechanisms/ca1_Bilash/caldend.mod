TITLE HVA (L-type) calcium channel with low threshold for activation (dendrites)

COMMENT
Used in distal dendrites to account for distally restricted initiation of Ca++ spikes.
Uses channel conductance (not permeability).
Written by Yiota Poirazi, 1/8/00 poirazi@LNC.usc.edu.
ENDCOMMENT

NEURON {
	SUFFIX caldend
	USEION ca READ eca WRITE ica
	RANGE gcalbar, ica
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
}

PARAMETER {	:parameters that can be entered when function is called in cell-setup
	gcalbar = 0 (S/cm2)	: initialized conductance
}

ASSIGNED {	:parameters needed to solve DE
	v (mV)
	celsius (degC)
	ica (mA/cm2)
	eca (mV)	: Ca2+ reversal potential
	minf (1)
	hinf (1)
	taum (ms)
	tauh (ms)
}

STATE {	:unknown activation and inactivation parameters to be solved in the DEs
	m
	h
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ica = gcalbar*pow(m, 3)*h*(v - eca)
}

DERIVATIVE states {
	rates (v)
	m' = (minf-m)/taum
	h' = (hinf-h)/tauh
}

INITIAL {
	rates(v)
	m = minf	:initial activation parameter value
	h = hinf	:initial inactivation parameter value
}

PROCEDURE rates(v (mV)) {
	minf = 1 / (1 + exp(-(v+37(mV))/1(mV)))		:Ca activation 
	taum = 3.6

	hinf = 1 / (1 + exp((v+41(mV))/0.5(mV)))	:Ca inactivation
	tauh = 29
}
