/*
This file is part of mfaktc.
Copyright (C) 2009, 2010  Oliver Weihe (o.weihe@t-online.de)

mfaktc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
                                
You should have received a copy of the GNU General Public License
along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
*/

__device__ static unsigned int __umul24hi(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm("mul24.hi.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __umul32(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm("mul.lo.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __umul32hi(unsigned int a, unsigned int b)
{
/*  unsigned int r;
  asm("mul.hi.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;*/
  return __umulhi(a,b);
}


__device__ static unsigned int __add_cc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm("add.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __addc_cc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm("addc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __addc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm("addc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __sub_cc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm("sub.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __subc_cc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm("subc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}


__device__ static unsigned int __subc(unsigned int a, unsigned int b)
{
  unsigned int r;
  asm("subc.u32 %0, %1, %2;" : "=r" (r) : "r" (a) , "r" (b));
  return r;
}
