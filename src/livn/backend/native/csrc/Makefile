# librcsd, the native livn backend.
#
# The flags are part of the contract: no -ffast-math (it reassociates and
# flushes denormals, so a seeded run stops reproducing across platforms) and
# no -march=native (the wheel has to run on the oldest x86-64 a consumer has).

CC ?= cc
CFLAGS ?= -O2
CFLAGS += -std=c99 -fno-fast-math -fPIC -Wall -Wextra -D_USE_MATH_DEFINES
LDLIBS ?= -lm

SRC := rcsd.c synapse.c noise.c opsin.c stimulus.c random123.c
HDR := rcsd.h internal.h mech.h

UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),Darwin)
    LIB ?= librcsd.dylib
    SHARED := -dynamiclib
else
    LIB ?= librcsd.so
    SHARED := -shared
endif

OUT ?= ..

all: $(OUT)/$(LIB)

$(OUT)/$(LIB): $(SRC) $(HDR)
	$(CC) $(CFLAGS) $(SHARED) -o $@ $(SRC) $(LDLIBS)

clean:
	rm -f $(OUT)/librcsd.so $(OUT)/librcsd.dylib $(OUT)/rcsd.dll

.PHONY: all clean
