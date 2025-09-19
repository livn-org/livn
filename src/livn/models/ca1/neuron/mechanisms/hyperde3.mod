TITLE hyperde3.mod  
 
COMMENT
Chen K, Aradi I, Thon N, Eghbal-Ahmadi M, Baram TZ, Soltesz I: Persistently
modified h-channels after complex febrile seizures convert the seizure-induced
enhancement of inhibition to hyperexcitability. Nature Medicine, 7(3) pp. 331-337, 2001.
(modeling by Ildiko Aradi, iaradi@uci.edu)
Distal dendritic Ih channel kinetics for both HT and Control animals

20190320 Aaron Milstein removed use of exotic ions, modern version of NEURON did not allow
previous semantic which used the same variable name for ions and state variables.
ENDCOMMENT
 
UNITS {
        (mA) =(milliamp)
        (mV) =(millivolt)
        (uF) = (microfarad)
	(molar) = (1/liter)
	(nA) = (nanoamp)
	(mM) = (millimolar)
	(um) = (micron)
	FARADAY = 96520 (coul)
	R = 8.3134	(joule/degC)
}
 
? interface 
NEURON { 
SUFFIX hyperde3
NONSPECIFIC_CURRENT i
RANGE ehyf, ehys
RANGE  ghyf, ghys
RANGE ghyfbar, ghysbar
RANGE hyfinf, hysinf, hyftau, hystau
RANGE ihyf, ihys
}
 
:INDEPENDENT {t FROM 0 TO 100 WITH 100 (ms)}
 
PARAMETER {

	ghyfbar = 0.000015 (mho/cm2)
	ghysbar = 0.000015 (mho/cm2)
	ehyf = -40. (mV)
	ehys = -40. (mV)
	ehyhtf = -40. (mV)
	ehyhts = -40. (mV)
}
 
STATE {
	hyf hys
}
 
ASSIGNED {
      v (mV)
      celsius (degC)

      ghyf (mho/cm2)
      ghys (mho/cm2)

      ihyf (mA/cm2)
      ihys (mA/cm2)
	  i    (mA/cm2)
      hyfinf
	  hysinf
      hyftau (ms)
	  hystau (ms)
} 

? currents
BREAKPOINT {

	SOLVE states METHOD cnexp

	ghyf = ghyfbar * hyf*hyf
	ihyf = ghyf * (v-ehyf)
	ghys = ghysbar * hys*hys
	ihys = ghys * (v-ehys)

	i = ihyf + ihys
    }
 
UNITSOFF
 
INITIAL {
	trates(v)
	
	hyf = hyfinf
	hys = hysinf
}

? states
DERIVATIVE states {	:Computes state variables m, h, and n 
        trates(v)	:      at the current v and dt.
        
        hyf' = (hyfinf-hyf) / hyftau
        hys' = (hysinf-hys) / hystau
}
 
? rates
PROCEDURE rates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
        LOCAL  alpha, beta, sum, q10
       q10 = 3^((celsius - 6.3)/10)
       
	:"hyf" FAST CONTROL Hype activation system
	hyfinf =  1 / (1 + exptrap(1, (v+91)/10 ))
	hyftau = 14.9 + 14.1 / (1+exptrap(2, -(v+95.2)/0.5))

	:"hys" SLOW CONTROL Hype activation system
	hysinf =  1 / (1 + exptrap(3, (v+91)/10 ))
	hystau = 80 + 172.7 / (1+exptrap(4, -(v+59.3)/-0.83))

}
 
PROCEDURE trates(v) {  :Computes rate and other constants at current v.
                      :Call once from HOC to initialize inf at resting v.
                           
	rates(v)	: not consistently executed from here if usetable_hh == 1
		: so don't expect the tau values to be tracking along with
		: the inf values in hoc

}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{  
                vtrap = x/(exptrap(0,x/y) - 1)
        }
}


FUNCTION exptrap(loc,x) {
  if (x>=700.0) {
    :printf("exptrap hyperde3 [%g]: x = %g\n", loc, x)
    exptrap = exp(700.0)
  } else {
    exptrap = exp(x)
  }
}
 
UNITSON

