TITLE Constant current for custom initialization

COMMENT
Taken from NEURON book, Chapter 8
ENDCOMMENT

NEURON {
	SUFFIX constant
	NONSPECIFIC_CURRENT i
	RANGE i, ic
}

UNITS {
	(mA) = (milliamp)
}

PARAMETER {
	ic = 0 (mA/cm2)
}

ASSIGNED {
	i (mA/cm2)
}

BREAKPOINT {
	i = ic
}
