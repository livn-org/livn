: Four state kinetic scheme for Potassium channel
: Based on A Numerical Approach to Ion Channel Modelling Using Whole-Cell Voltage-Clamp Recordings and a Genetic Algorithm.
: Meron Gurkiewicz and Alon Korngreen. 
: PLoS Comput Biol. 2007 Aug; 3(8): e169.

NEURON {
      SUFFIX K_GK
      USEION k READ ek WRITE ik
      RANGE g, gbar, a12, a21, z12, z21
}
UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

PARAMETER {
      gbar = 0     (S/cm2)
      a12 = 0.01   (/ms)
      a21 = 0.02   (/ms)
      a23 = 0.01   (/ms)
      a32 = 0.02   (/ms)
      z12 = 0.01   (/mV)
      z21 = 0.02   (/mV)
      z23 = 0.01   (/mV)
      z32 = 0.02   (/mV)
}

ASSIGNED {
      v    (mV)
      ek   (mV)
      g    (pS/um2)
      ik   (mA/cm2)
      k12  (/ms)
      k21  (/ms)
      k23  (/ms)
      k32  (/ms)
}

STATE { c1 c2 o }

BREAKPOINT {
      SOLVE states METHOD sparse
      g = gbar*o
      ik = g*(v - ek)
}

INITIAL { SOLVE states STEADYSTATE sparse} 

KINETIC states {   		
        rates(v)
	~c1 <-> c2(k12,k21)
	~c2 <-> o(k23,k32)
	CONSERVE c1+c2+o=1
}

PROCEDURE rates(v(millivolt)) {

      k12 = a12*exp(z12*v)
      k21 = a21*exp(-z21*v)
      k23 = a23*exp(z23*v)
      k32 = a32*exp(-z32*v)
}
