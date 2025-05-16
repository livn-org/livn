: Na channel


NEURON {
	SUFFIX Na_PR
	USEION na READ ena WRITE ina
	RANGE gmax, g
        GLOBAL ahk, ahs, ahd, bhk, bhs, bhd
        GLOBAL amk, ams, amd, bmk, bms, bmd
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
    ahs = 0.128
    ahk = -43
    ahd = 18
    bhs = 4.
    bhk = -20
    bhd = 5
    ams = 0.32
    amk = -46.9
    amd = 4
    bms = 0.28
    bmk = 19.9
    bmd = 5
    
}

ASSIGNED {
	v		(mV)
	m_inf		(mV)
        m_tau (ms)
	ena		(mV)
	ina		(mA/cm2)
        g  	        (S/cm2)
        ah	        (/ms)	
	bh		(/ms)	
        am	        (/ms)	
	bm		(/ms)	
        celsius (degC)
}

STATE { h }

INITIAL { 
	rates(v)
        h=ah/(ah + bh)
        m_inf=am/(am + bm)
}
    
BREAKPOINT {
    SOLVE states METHOD cnexp
    g = gmax*(m_inf^2)*h
    ina = g*(v - ena)
} 

DERIVATIVE states {
   rates(v)
   h' = ah - h*(ah + bh)
}


PROCEDURE rates(v (mV)) { LOCAL tcorr
    
    tcorr = Q10^((celsius - 36(degC))/10 (degC))
    
    am=tcorr*ams*linoid(amk-v, amd)
    bm=tcorr*bms*linoid(v+bmk, bmd)
    m_inf=am/(am + bm)
    
    ah=tcorr*ahs*exp((ahk-v)/ahd)
    bh=tcorr*bhs/(1+exp((bhk-v)/bhd))
}

 
FUNCTION linoid(x,y) {
    if (fabs(x/y) < 1e-6) {
        linoid = y*(1 - x/y/2)
    } else {
        linoid = x/(exp(x/y) - 1)
    }
}
