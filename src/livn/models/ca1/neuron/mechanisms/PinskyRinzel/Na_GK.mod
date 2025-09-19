: Na channel
: Based on A Numerical Approach to Ion Channel Modelling Using Whole-Cell Voltage-Clamp Recordings and a Genetic Algorithm.
: Meron Gurkiewicz and Alon Korngreen. 
: PLoS Comput Biol. 2007 Aug; 3(8): e169.



NEURON {
	SUFFIX Na_GK
	USEION na READ ena WRITE ina
	RANGE gbar, g, am1, am2, am3, bm1, bm2, ah1, ah2, bh1, bh2, bh3
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
}

PARAMETER {
  gbar = 0 (S/cm2)
  am1 = 10
  am2 = 0.1
  am3 = 0.2
  bm1 = 0.1
  bm2 = 10
  ah1 = 0.1
  ah2 = 10
  bh1 = 1
  bh2 = 0.2
  bh3 = 10
}

ASSIGNED {
	v		(mV)
	ena		(mV)
	ina		(mA/cm2)
        g  	        (S/cm2)
        am
        bm
        ah
        bh
        
}

STATE { m h }

INITIAL { 
    rates(v)
    m=am/(am + bm)
    h=ah/(ah + bh)
}
    
BREAKPOINT {
    rates(v)
    SOLVE states METHOD cnexp
    g = gbar*m^3*h
    ina = g*(v - ena)
} 

DERIVATIVE states {
   rates(v)
   m' = am*(1 - m) - bm*m
   h' = ah*(1 - h) - bh*h
}


PROCEDURE rates(v (mV)) { 
    am=am2*(v - am1)/(1 - exp(-am3*(v - am1)))
    bm=bm1*exp(-v/bm2)
    
    ah=ah1*exp(-v/ah2)
    bh=bh1/(1 + exp(-bh2 * (v + bh3)))
}