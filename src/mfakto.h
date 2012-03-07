/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)

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
*/

#ifndef mfakto_H_
#define mfakto_H_

#define NUM_KERNELS 15
#define KERNEL_FILE "mfakto_Kernels.cl"

#ifdef __cplusplus
extern "C"
{
#endif

int init_CL(int num_streams, cl_uint devicenumber);
int init_CLstreams(void);
int cleanup_CL(void);
void CL_test(cl_uint devicenumber);
int tf_class_opencl(unsigned int exp, int bit_min, unsigned long long int k_min,
   unsigned long long int k_max, mystuff_t *mystuff, enum GPUKernels use_kernel);

#ifdef __cplusplus
}
#endif

int run_kernel(cl_kernel l_kernel, cl_uint exp, int stream, cl_mem res);
int run_mod_kernel(cl_ulong hi, cl_ulong lo, cl_ulong q, cl_float qr, cl_ulong *res_hi, cl_ulong *res_lo);


#endif  /* #ifndef mfakto_H_ */
