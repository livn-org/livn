: K channel


NEURON {
	SUFFIX K_PR
	USEION k READ ek WRITE ik
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
        an	        (/ms)	
	bn		(/ms)	
        ninf
        ntau (ms)
        celsius (degC)
}

STATE { n }

INITIAL { 
	rates(v)
	n = ninf
}
    
BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gmax * n
    ik = g*(v - ek)
} 

DERIVATIVE states {
   rates(v)
   n' = (ninf - n)/ntau
  
}


PROCEDURE rates(v (mV)) { LOCAL tcorr, sum
    
    tcorr = Q10^((celsius - 36(degC))/10 (degC))
    
    an=tcorr*0.016*linoid(-24.9-v, 5)
    bn=tcorr*0.25*exp(-1-0.025*v)
    
    sum = an + bn
    ninf = an/sum
    ntau = 1/sum
}

 
FUNCTION linoid(x,y) {
    if (fabs(x/y) < 1e-6) {
        linoid = y*(1 - x/y/2)
    } else {
        linoid = x/(exp(x/y) - 1)
    }
}
