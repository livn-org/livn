TITLE Calcium Pumping/Buffering

COMMENT
Internal calcium concentration due to calcium currents and pump.
Differential equations.

Decay of internal calcium concentration

The mechanism is taken from: Destexhe et al., 1994, J Comput Neurosci, 1:195-230, doi: 10.1007/BF00961734
This mechanism was published in Destexhe et al., 1993, Biophys J, 65: 1538-1552, doi: 10.1016/S0006-3495(93)81190-1
Written by Alain Destexhe, Salk Institute, Nov 12, 1992.

Model:
	Simple model of ATPase pump with 3 kinetic constants (Destexhe 92)
	Cai + P <-> CaP -> Cao + P  (k1,k2,k3)
	A Michaelis-Menten approximation is assumed, which reduces the complexity
	of the system to 2 parameters

		kt = <tot enzyme concentration> * k3  -> TIME CONSTANT OF THE PUMP
		kd = k2/k1 (dissociation constant)    -> EQUILIBRIUM CALCIUM VALUE

The values of these parameters are chosen assuming a high affinity of 
the pump to calcium and a low transport capacity (cfr. Blaustein et al., 1988,
Trends Neurosci, 11:438-443, doi: 10.1016/0166-2236(88)90195-6, and references therein).  

Units checked using "modlunit" -> factor 10000 needed in ca entry.

VERSION OF PUMP + DECAY (decay can be viewed as simplified buffering).

All variables are range variables.

This file was modified by Yiota Poirazi (poirazi@LNC.usc.edu) on April 18, 2001 to account for the sharp
Ca++ spike repolarization observed
Ref: Golding et al., 1999, J Neurosci, 19: 8789-8798, doi: 10.1523/JNEUROSCI.19-20-08789.1999

factor 10000 is replaced by 10000/18 needed in ca entry
ENDCOMMENT

NEURON {
	SUFFIX cad
	USEION ca READ ica, cai WRITE cai
	RANGE ca, ica, cai
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(um) = (micron)
	(mA) = (milliamp)
	FARADAY = (faraday) (coulomb)
}

PARAMETER {
	depth = .1 (um)		: depth of shell
	taur = 200 (ms)		: rate of calcium removal
	cainf = 100e-6 (mM)
}

ASSIGNED {
	cai	(mM)
	ica	(mA/cm2)
	drive_channel (mM/ms)
}

STATE { ca (mM) }
	
BREAKPOINT {
	SOLVE state METHOD derivimplicit
}

DERIVATIVE state { 
	drive_channel =  - (1e4) * ica / (2 * FARADAY * depth)
	
	if (drive_channel <= 0.) { drive_channel = 0. }		: cannot pump inward
	
	ca' = drive_channel/18 + (cainf - ca)/taur
	cai = ca
}

INITIAL { ca = cainf }
