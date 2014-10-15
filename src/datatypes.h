/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2014  Oliver Weihe (o.weihe@t-online.de)
                           Bertram Franz (bertramf@gmx.net)

mfaktc (mfakto) is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc (mfakto) is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with mfaktc (mfakto).  If not, see <http://www.gnu.org/licenses/>.

Version 0.15

*/

/* all datatypes used by the various kernels */

#ifdef MORE_CLASSES
#define NUM_CLASSES 4620u
#else
#define NUM_CLASSES 420u
#endif

#define CON2(a,b) a##b
#define CONC(a,b) CON2(a,b)

/* 96bit (3x 32bit) integer
D= d0 + d1*(2^32) + d2*(2^64) */
typedef struct _int96_t
{
  uint d0,d1,d2;
}int96_t;

/* 192bit (6x 32bit) integer
D=d0 + d1*(2^32) + d2*(2^64) + ... */
typedef struct _int192_t
{
  uint d0,d1,d2,d3,d4,d5;
}int192_t;

/* 72bit (3x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) */
typedef struct _int72_t
{
  uint d0,d1,d2;
}int72_t;

/* 144bit (6x 24bit) integer
D=d0 + d1*(2^24) + d2*(2^48) + ... */
typedef struct _int144_t
{
  uint d0,d1,d2,d3,d4,d5;
}int144_t;

// 5x15bit
typedef struct _int75_t
{
  uint d0,d1,d2,d3,d4;
}int75_t;

// 10x15bit
typedef struct _int150_t
{
  uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int150_t;

// 6x15bit
typedef struct _int90_t
{
  uint d0,d1,d2,d3,d4,d5;
}int90_t;

// 12x15bit
typedef struct _int180_t
{
  uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db;
}int180_t;

// 5x16bit
typedef struct _int80_t
{
  uint d0,d1,d2,d3,d4;
}int80_t;

// 10x16bit
typedef struct _int160_t
{
  uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int160_t;

////////// vectorized data types /////////////

#if (VECTOR_SIZE == 1)
typedef struct _int72_v
{
  uint d0,d1,d2;
}int72_v;

typedef struct _int144_v
{
  uint d0,d1,d2,d3,d4,d5;
}int144_v;

typedef struct _int96_v
{
  uint d0,d1,d2;
}int96_v;

typedef struct _int192_v
{
  uint d0,d1,d2,d3,d4,d5;
}int192_v;

typedef struct _int75_v
{
  uint d0,d1,d2,d3,d4;
}int75_v;

typedef struct _int150_v
{
  uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int150_v;

typedef struct _int90_v
{
  uint d0,d1,d2,d3,d4,d5;
}int90_v;

typedef struct _int180_v
{
  uint d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db;
}int180_v;

#define int_v int
#define uint_v uint
#define ulong_v ulong
#define float_v float
#define CONVERT_FLOAT_V convert_float
#define CONVERT_FLOAT_RTP_V convert_float
#define double_v double
#define CONVERT_DOUBLE_V convert_double
#define CONVERT_DOUBLE_RTP_V convert_double
#define CONVERT_UINT_V convert_uint
#define CONVERT_ULONG_V convert_ulong
// AS_UINT is applied only to logical results. For vector operations, these are 0 (false) or -1 (true)
// For scalar operations, they result in 0 (false) or 1 (true) ==> to unify, negate here
#define AS_INT_V(x) as_int((x)?-1:0)
#define AS_LONG_V(x) as_long((x)?-1:0)
#define AS_UINT_V(x) as_uint((x)?-1:0)
#define AS_ULONG_V(x) as_ulong((x)?-1:0)
// to unify printf's:
#define V(x) x
#else
typedef struct _int72_v
{
  CONC(uint,VECTOR_SIZE) d0,d1,d2;
}int72_v;

typedef struct _int144_v
{
  CONC(uint,VECTOR_SIZE) d0,d1,d2,d3,d4,d5;
}int144_v;

typedef struct _int96_v
{
  CONC(uint,VECTOR_SIZE) d0,d1,d2;
}int96_v;

typedef struct _int192_v
{
  CONC(uint,VECTOR_SIZE) d0,d1,d2,d3,d4,d5;
}int192_v;

typedef struct _int75_v
{
  CONC(uint,VECTOR_SIZE) d0,d1,d2,d3,d4;
}int75_v;

typedef struct _int150_v
{
  CONC(uint,VECTOR_SIZE) d0,d1,d2,d3,d4,d5,d6,d7,d8,d9;
}int150_v;

typedef struct _int90_v
{
  CONC(uint,VECTOR_SIZE) d0,d1,d2,d3,d4,d5;
}int90_v;

typedef struct _int180_v
{
  CONC(uint,VECTOR_SIZE) d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,da,db;
}int180_v;

#define int_v CONC(int,VECTOR_SIZE)
#define uint_v CONC(uint,VECTOR_SIZE)
#define ulong_v CONC(ulong,VECTOR_SIZE)
#define float_v CONC(float,VECTOR_SIZE)
#define double_v CONC(double,VECTOR_SIZE)
// _rtp/_rtz are MUCH slower than the default (at least on HD5770)
//#define CONVERT_FLOAT_V CONC(CONC(convert_float,VECTOR_SIZE), _rtz)
//#define CONVERT_FLOAT_RTP_V CONC(CONC(convert_float,VECTOR_SIZE), _rtp)
//#define CONVERT_UINT_V CONC(CONC(convert_uint,VECTOR_SIZE), _rtz)
#define CONVERT_FLOAT_V CONC(convert_float,VECTOR_SIZE)
#define CONVERT_FLOAT_RTP_V CONC(convert_float,VECTOR_SIZE)
#define CONVERT_DOUBLE_V CONC(convert_double,VECTOR_SIZE)
#define CONVERT_DOUBLE_RTP_V CONC(convert_double,VECTOR_SIZE)
#define CONVERT_UINT_V CONC(convert_uint,VECTOR_SIZE)
#define CONVERT_ULONG_V CONC(convert_ulong,VECTOR_SIZE)
#define AS_INT_V CONC(as_int,VECTOR_SIZE)
#define AS_LONG_V CONC(as_long,VECTOR_SIZE)
#define AS_UINT_V CONC(as_uint,VECTOR_SIZE)
#define AS_ULONG_V CONC(as_ulong,VECTOR_SIZE)
// to unify printf's:
#define V(x) x.s0
#endif

// define to efficiently handle carry/borrow
// ADD_COND returns val+1 if cond is true, otherwise val
// SUB_COND returns val-1 if cond is true, otherwise val
#if defined VLIW4 || defined VLIW5
// VLIW4/5 native instructions already return -1 on vector "true": use it directly
#define ADD_COND(val, cond) (val - AS_UINT_V(cond))
#define SUB_COND(val, cond) (val + AS_UINT_V(cond))
#else
// GCN (and others) don't really know vectors and return 1 for "true" in their native instructions
// use this define to allow the optimizer to circumvent the OpenCL convention to return -1
#define ADD_COND(val, cond) (val + AS_UINT_V((cond) ? 1 : 0))
#define SUB_COND(val, cond) (val - AS_UINT_V((cond) ? 1 : 0))
#endif

#define CONVERT_FLOAT convert_float
#define CONVERT_FLOAT_RTP convert_float
#define CONVERT_DOUBLE convert_double
#define CONVERT_DOUBLE_RTP convert_double

