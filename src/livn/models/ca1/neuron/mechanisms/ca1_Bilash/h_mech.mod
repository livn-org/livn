TITLE hyperpolarization-activated channel - h current

COMMENT
Used in Pyramidal Cells.
Based on Magee, 1988, J Neurosci, 18:7613-7624, doi: 10.1523/JNEUROSCI.18-19-07613.1998
ENDCOMMENT

NEURON {
	SUFFIX h_mech
	NONSPECIFIC_CURRENT ih
	RANGE  ghbar, vhalf, eh, ih
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
}

PARAMETER {: parameters that can be entered when function is called in cell-setup
	ghbar = 0.0 (S/cm2)  : initialize conductance to zero
	eh = -10 (mV)
	K = 3.5 (mV)
	vhalf = -90 (mV)    : half potential
}

ASSIGNED {: parameters needed to solve DE
	v (mV)
	ih (mA/cm2)
	ninf
	taun (ms)
}

STATE {: the unknown parameters to be solved in the DEs
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ih = ghbar*n*(v-eh)
}

DERIVATIVE states {
	rates(v)
	n' = (ninf - n)/taun
}

INITIAL {: initialize the following parameter using states()
	rates(v)
	n = ninf
}

PROCEDURE rates(v (mV)) {
	if (v > -30) {
		taun = 1
	} else {
		taun = 2(ms)*(1/(exp(-(v+145(mV))/17.5(mV)) + exp((v+16.8(mV))/16.5(mV))) + 10) :h activation tau
	}
	ninf = (1 / (1 + exp((v - vhalf)/K)))	:steady state value
}
