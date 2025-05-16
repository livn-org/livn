: K channel


NEURON {
	SUFFIX KCa_PR
	USEION k READ ek WRITE ik
        USEION ca READ cai
	RANGE gmax, g
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
    Q10 = 3 (1)
}

ASSIGNED {
	v		(mV)
	ek		(mV)
	ik		(mA/cm2)
        g  	        (S/cm2)
        cinf	        (/ms)	
	tauc		(ms)	
        cai (mM)
        celsius (degC)
        chid
}

STATE { c }

INITIAL { 
	rates(v)
        c=cinf
}
    
BREAKPOINT {
    SOLVE states METHOD derivimplicit
    g = gmax * c * chid
    ik = g*(v-ek)
} 

DERIVATIVE states {
   rates(v)
   c'=(cinf-c)/tauc
}


PROCEDURE rates(v (mV)) { LOCAL tcorr
    
    tcorr = Q10^((celsius - 36(degC))/10 (degC))

    cinf=expit((10.1+v)/0.1016)^0.00925
    tauc=3.627*exp(0.03704*v)/tcorr
    chid = (1.073*sin(0.003453*cai+0.08095) + 0.08408*sin(0.01634*cai-2.34) + 0.01811*sin(0.0348*cai-0.9918))
    
}

FUNCTION expit(x) { LOCAL a
    if (x < 0) {
        a = exp(x)
        expit = a / (1 + a)
    } else {
      expit = 1/(1 + exp(-x))
    }
}


