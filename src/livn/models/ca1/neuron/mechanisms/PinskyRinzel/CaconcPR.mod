

NEURON {
	SUFFIX Ca_conc_PR
	USEION ca READ ica WRITE cai
	RANGE cai0, cai, d, beta, irest
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mV) = (millivolt)
	(mA) = (milliamp)
	F = (faraday) (coulomb)
        R = 8.3134      (joule/degC)
}

PARAMETER {
	cai0 = 1e-5 	(mM)
	:cao0 = 10.0 	(mM)
        d = 13 (mM)
        beta = 0.075       (/ms)
}

ASSIGNED {
	v			(mV)
	ica			(mA/cm2)
	irest  (mA/cm2)
        celsius (degC)
    }

STATE {
	cai (mM) <1e-5>
        :cao (mM)
    }
    
BREAKPOINT { 
    SOLVE state METHOD derivimplicit
}

DERIVATIVE state { LOCAL channel_flow
    
    channel_flow = -d*10*(ica - irest)
    if (channel_flow < 0) {
        channel_flow = 0
    }
    cai' = channel_flow - beta*(cai - cai0)
    :cao' = -channel_flow - beta*(cao - cao0)
}


INITIAL {
    cai = cai0
    irest = ica
    :cao = cao0
}
