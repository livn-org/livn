INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    POINT_PROCESS Ifluct1
    RANGE m, s, tau, x
    RANGE new_seed
    NONSPECIFIC_CURRENT i
    THREADSAFE : only true if every instance has its own distinct Random
    POINTER randObjPtr
}


UNITS {
    (nA) = (nanoamp) 
    (mV) = (millivolt)
}


PARAMETER {
    m   = 0. (nA)      : steady-state expected value of the current amplitude
    s   = 0. (nA)      : square root of the steady-state variance of the current amplitude
    tau = 2. (ms)      : steady-state correlation time length of the current
}

ASSIGNED {
    noise
    i     (nA)          : fluctuating current
    x                   : state variable
    randObjPtr
}

INITIAL {
    x = m               : to reduce the transient, the state is set to its (expected) steady-state
}


BEFORE BREAKPOINT {
    noise = randGen()
}

BREAKPOINT {
    SOLVE oup
    if (tau <= 0) {  x = m + s  * noise }  : white-noise is impossible to generate anyway..
    i = - x
}

PROCEDURE oup() {       
    if (tau > 0) {  x = x + (1. - exp(-dt/tau)) * (m - x) + sqrt(1.-exp(-2.*dt/tau)) * s  * noise }
}

VERBATIM
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
ENDVERBATIM

FUNCTION randGen() {
VERBATIM
   if (_p_randObjPtr) {
      /*
      :Supports separate independent but reproducible streams for
      : each instance. However, the corresponding hoc Random
      : distribution MUST be set to Random.normal(0,1)
      */
      _lrandGen = nrn_random_pick(_p_randObjPtr);
   }else{
      hoc_execerror("Random object ref not set correctly for randObjPtr"," only via hoc Random");
   }
ENDVERBATIM
}

PROCEDURE setRandObj() {
VERBATIM
   void** pv4 = (void**)(&_p_randObjPtr);
   if (ifarg(1)) {
      *pv4 = nrn_random_arg(1);
   }else{
      *pv4 = (void*)0;
   }
ENDVERBATIM
}

