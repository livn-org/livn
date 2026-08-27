TITLE Sodium Channel

NEURON {
	SUFFIX Nas
	USEION na READ ena WRITE ina
	RANGE gmax, ina, g, vhalf, slope
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S)  = (siemens)
}

PARAMETER {
	gmax	=0.06 	(mho/cm2) <0,1e9>
	vhalf	=-35	(mV)
	slope	=7.8	(mV) <1e-9,1e9>
}

ASSIGNED {
	v (mV)
	ena (mV)
	ina (mA/cm2)
	g (S/cm2)
	minf
	hinf htau
}

STATE {
	h
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	g = gmax * minf^3 * h
	ina = g * (v - ena)
}

INITIAL { 
    rates(v)
    h = hinf
}

DERIVATIVE states { 
	rates(v)
	h' = (hinf - h)/htau
}

PROCEDURE rates(v(mV)) {LOCAL a, b
        minf = 1/(1+exp(-(v-vhalf)/slope))
	htau = 30/(exp((v+50)/15)+exp(-(v+50)/16))
	hinf = 1/(1+exp((v+55)/7))
}