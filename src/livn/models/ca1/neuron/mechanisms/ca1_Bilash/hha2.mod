TITLE HH channel that includes both a sodium and a delayed rectifier channel

COMMENT
Original taken from: Bernader, et al., 1991, Proc. Natl. Acad. Sci. USA 88, 11569-11573, doi: 10.1073/pnas.88.24.11569
Accounts for sodium conductance attenuation:
	Jung et al., 1997, J. Neurosci. 17, 6639-6647, doi: 10.1523/JNEUROSCI.17-17-06639.1997
	Migliore et al., 1999, J. Comput. Neurosci. 7, 5-15, doi: 10.1023/a:1008906225285

Bartlett Mel-modified Hodgkin - Huxley conductances
Terrence Brannon-added attenuation 
Yiota Poirazi-modified Kdr and Na threshold and time constants to make it more stable
Yiota Poirazi-modified threshold for soma/axon spike initiation (threshold about -57 mV), USC Los Angeles 2000, poirazi@LNC.usc.edu
This file is used ONLY in soma and axon sections
ENDCOMMENT

NEURON {
	SUFFIX hha2
	USEION na READ ena WRITE ina
	USEION k READ ek WRITE ik
	NONSPECIFIC_CURRENT il
	RANGE gnabar, gkbar, gl, el, ik, il, ina
	RANGE ar2
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
	FARADAY = (faraday) (kilocoulombs)
	R = (k-mole) (joule/degC)
}

PARAMETER {
	: parameters that can be entered when function is called in cell-setup
	a0r = 0.0003 (/ms)
	b0r = 0.0003 (/ms)
	zetar = 12
	zetas = 12
	gmr = 0.2
	: Na-conductance attenuation, "s", (ar=1 -> zero attenuation)
	ar2 = 1.0			:initialized parameter for location-dependent
	taumin = 3 (ms)		:min activation time for "s" attenuation system
	vvs = 2 (mV)		:slope for "s" attenuation system
	vhalfr = -50 (mV)	:half potential for "s" attenuation system

	gnabar = 0 (S/cm2)
	gkbar = 0 (S/cm2)
	gl = 0 (S/cm2)
}

ASSIGNED { : parameters needed to solve DE
	v (mV)
	ena (mV)
	ek (mV)
	el (mV)
	celsius (degC)

	ina (mA/cm2)
	ik (mA/cm2)
	il (mA/cm2)

	minf
	taum (ms)
	hinf
	tauh (ms)
	sinf
	taus (ms)
	ninf
	taun (ms)
}

STATE { :the unknown parameters to be solved in the DEs
	m
	h
	s
	n
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ina = gnabar*pow(m, 2)*h*s*(v - ena)	:Sodium current
	ik = gkbar*pow(n, 2)*(v - ek)			:Potassium current
	il = gl*(v - el)				:leak current
}

DERIVATIVE states {
	rates(v)
	m' = (minf - m)/taum :Na activation variable
	h' = (hinf - h)/tauh :Na inactivation variable
	s' = (sinf - s)/taus :Na attenuation variable
	n' = (ninf - n)/taun :K activation variable
}

INITIAL {:initialize the following parameter using states()
	rates(v)
	m = minf
	h = hinf
	s = 1
	n = ninf
}

FUNCTION alpr(v (mV)) (1) {
	:used in "s" activation system tau
	alpr = exp(1.e-3*zetar*(v-vhalfr)*FARADAY/(R*(273.16+celsius)))
}

FUNCTION betr(v (mV)) (1) {
	:used in "s" activation system tau
	betr = exp(1.e-3*zetar*gmr*(v-vhalfr)*FARADAY/(R*(273.16+celsius)))
}

PROCEDURE rates(v (mV)) {
	minf = 1 / (1 + exp((v - 3.0(mV) + 42.5(mV))/(-3(mV))))		: Na activation
	hinf = 1 / (1 + exp((v - 3.0(mV) + 49(mV))/(3.5(mV))))		: Na inactivation 
	sinf = (1+ar2*exp((v-vhalfr)/vvs))/(1+exp((v-vhalfr)/vvs))	: Na attenuation
	ninf = 1 / (1 + exp((v - 3.0(mV) + 46.3(mV))/(-3(mV))))		: K activation

	taum = 0.05	:Na activation tau
	tauh = 1.0	:Na inactivation tau
	taun = 1.0 :3.5	:K activation tau

	taus = betr(v)/(a0r+b0r*alpr(v)) :s activation tau
	if (taus<taumin) {taus = taumin}
}
