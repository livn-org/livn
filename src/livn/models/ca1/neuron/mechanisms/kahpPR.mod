: K channel


NEURON {
	SUFFIX KAHP_PR
	USEION k READ ek WRITE ik
        USEION ca READ cai
	RANGE gmax, g, bq, aqs, aqmax
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
}

PARAMETER {
    gmax = 0 (S/cm2)
    bq = 0.001
    aqs = 0.00002
    aqmax = 0.01
}

ASSIGNED {
	v		(mV)
	ek		(mV)
	ik		(mA/cm2)
        g  	        (S/cm2)
        qinf	        (/ms)	
	tauq		(ms)	
        cai (mM)
        celsius (degC)
        aq
}

STATE { q }

INITIAL { 
    rates(v)
    q=qinf
}
    
BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gmax * q
    ik = g*(v - ek)
} 

DERIVATIVE states {
   rates(v)
   q' = (qinf - q)/tauq
}


PROCEDURE rates(v (mV)) { 
    
    aq  = aqs * cai
    if (aqmax < aq) {
        aq = aqmax
    }
    qinf = aq/(aq + bq)
    tauq = 1/(aq + bq)
    
}    




