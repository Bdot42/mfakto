# This file is part of mfaktc (mfakto).
# Copyright (C) 2009 - 2011, 2014  Oliver Weihe (o.weihe@t-online.de)
#                                  Bertram Franz (bertramf@gmx.net)
#
# mfaktc (mfakto) is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# mfaktc (mfakto) is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mfaktc (mfakto).  If not, see <http://www.gnu.org/licenses/>.
#
# Version 0.15
#
#
# Example: "make bitness=32 static=yes" to compile for 32 bits and link
#          statically

# set x86_64 (64-bit) as default
ARCH := x86_64
BITS := -m64

# check whether to build 32-bit version
ifdef bitness
  ifeq ($(bitness), 32)
    ARCH := x86
    BITS := -m32
  endif
endif

# Check if static linking is requested
ifdef static
  ifeq ($(static), yes)
    STATIC := -static
  endif
endif

# use OS-specific commands
ifneq (, $(shell which install))
  CP = install -m 555
  RM = rm -f
else
  ifeq ($(OS), Windows_NT)
    CP = copy
    RM = del /f
  else
    CP = cp
    RM = rm -f
  endif
endif

# where is the OpenCL SDK installed?
AMD_APP_DIR = /opt/rocm/opencl
AMD_APP_INCLUDE = -I$(AMD_APP_DIR)/include
AMD_APP_LIB = -L$(AMD_APP_DIR)/lib/$(ARCH)

# Change needed for compilation with amdgpu-pro
# AMD_APP_DIR = /opt/amdgpu-pro/opencl

# optimize or debug
#OPTIMIZE_FLAG = -O3
#OPTIMIZE_FLAG = -O3 -funroll-loops  -finline-functions -frerun-loop-opt -fgcse-sm -fgcse-las
OPTIMIZE_FLAG = -O3 -funroll-loops  -ffast-math -finline-functions -frerun-loop-opt -fgcse-sm -fgcse-las -flto
#OPTIMIZE_FLAG = -g
#OPTIMIZE_FLAG = -ggdb

# compiler settings for C and C++ files
CC = gcc
CPP = $(CC)
CFLAGS = $(BITS) -Wall $(OPTIMIZE_FLAG) $(AMD_APP_INCLUDE) -DBUILD_OPENCL
CPPFLAGS =
#CFLAGS_EXTRA_SIEVE = -funroll-all-loops
#CFLAGS_EXTRA_SIEVE = -funroll-all-loops -funsafe-loop-optimizations -fira-region=all -fsched-spec-load -fsched-stalled-insns=10 -fsched-stalled-insns-dep=10 -floop-parallelize-all -fvariable-expansion-in-unroller -fno-align-labels
CFLAGS_EXTRA_SIEVE = -funroll-all-loops -funsafe-loop-optimizations -fira-region=all -fsched-spec-load -fsched-stalled-insns=10 -fsched-stalled-insns-dep=10 -fno-align-labels

# Linker
LD = g++
LDFLAGS = $(BITS) $(STATIC) $(OPTIMIZE_FLAG) $(AMD_APP_LIB) -lOpenCL

##############################################################################

CSRC = sieve.c timer.c parse.c read_config.c mfaktc.c checkpoint.c \
	signal_handler.c filelocking.c output.c

# CLSRC = barrett15.cl  barrett.cl  common.cl  gpusieve.cl  mfakto_Kernels.cl  montgomery.cl  mul24.cl

COBJS = $(CSRC:.c=.o) mfakto.o gpusieve.o perftest.o menu.o kbhit.o

##############################################################################

.PHONY all:
all: ../mfakto ../barrett15.cl  ../barrett.cl  ../common.cl  ../gpusieve.cl  ../mfakto_Kernels.cl  ../montgomery.cl  ../mul24.cl ../datatypes.h ../tf_debug.h ../mfakto.ini
	#@echo OS is $(OS) and ARCH is $(ARCH) and SHELL is $(SHELL)
	echo $@ > $@

../mfakto : $(COBJS)
	$(LD) $^ $(LDFLAGS) -o $@

.PHONY clean:
clean:
	$(RM) *.o *~ depend

sieve.o : sieve.c
	$(CC) $(CFLAGS) $(CFLAGS_EXTRA_SIEVE) -c $< -o $@

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o : %.cpp
	$(CPP) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

../%.cl : %.cl
	$(CP) $< ..

../%.h : %.h
	$(CP) $< ..

../%.ini : %.ini
	$(CP) $< ..

##############################################################################

depend: $(CSRC)
	$(CC) -MM $(CFLAGS) $(CSRC) > $@.tmp
	mv $@.tmp $@

ifneq ($(MAKECMDGOALS), clean)
include depend
endif