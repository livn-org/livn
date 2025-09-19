:slow time constant introduced

NEURON {
  SUFFIX KA
  USEION k READ ek WRITE ik
  RANGE gbar, g, i
}

UNITS {
  (S) = (siemens)
  (mV) = (millivolt)
  (mA) = (milliamp)	
}

PARAMETER {
  gbar = 0.0 (S/cm2)
  btau = 235 (ms)
  atau = 12 (ms)
}

ASSIGNED {
  v	(mV)
  ek	(mV)
  ik 	(mA/cm2)
  i 	(mA/cm2)
  g	(S/cm2)
  
 
}

STATE { a b }


BREAKPOINT {
  SOLVE states METHOD cnexp
  g = gbar*a*a*a*a*b
  i = g*(v-ek)
  ik = i
}
  
INITIAL {
  b = binf(v)
  a = ainf(v)
}

DERIVATIVE states {
 b'= (binf(v)-b)/btau
 a' = (ainf(v)-a)/atau
}

FUNCTION ainf (Vm (mV)) () {

  UNITSOFF
    ainf = 1/(1+exp(-(Vm+60)/10))
  UNITSON

}


FUNCTION binf (Vm (mV)) () {

  UNITSOFF
    binf = 1/(1+exp((Vm+80)/6))
  UNITSON

}


