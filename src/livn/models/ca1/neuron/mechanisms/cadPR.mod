: Ca channel


NEURON {
	SUFFIX Ca_PR
	USEION ca READ cai,cao WRITE ica
	RANGE g, gmax
}

UNITS {
	(molar) = (1/liter)
	(mM) = (millimolar)
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
	F = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
}

PARAMETER {
    gmax = 0	(S/cm2)	: maximum permeability
    Q10 = 3 (1)

}
    
ASSIGNED {
	v		(mV)
	ica		(mA/cm2)
	g		(S/cm2)
        as
        bs
        ar
        br
        celsius (degC)
        cai (mM)
        cao (mM)
}

STATE { s r }

BREAKPOINT {
    SOLVE state METHOD derivimplicit
    g = gmax*(s^2)*r
    ica = g*ghk(v, cai, cao)
}

: ghk formalism from cal.mod
FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {
        LOCAL nu,f

        f = KTF(celsius)/2
        nu = v/f
        ghk=-f*(1. - (ci/co)*exp(nu))*efun(nu)
}

FUNCTION KTF(celsius (degC)) (mV) {
        KTF = ((36./293.15)*(celsius + 273.15))
}


FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}
    
DERIVATIVE state {	
    rates(v)
    s' = (as - s*(as + bs))
    r' = (ar - r*(ar + br))
}

INITIAL {
    rates(v)
    s = as/(as + bs)
    r = ar/(ar + br)
}


PROCEDURE rates(v (mV)) { LOCAL tcorr, xs
    
    tcorr = Q10^((celsius - 36(degC))/10 (degC))
    
    as = tcorr*5/(1 + exp(0.1*(5 - v)))
    xs = -0.2*(v + 8.9)
    bs = tcorr*0.2*xs/(1 - exp(-xs))
    
    ar = tcorr*0.1673*exp(-0.03035*(v + 38.5))
    br = tcorr*0.5/(1 + exp(0.3*(8.9 - v)))
    
}
