#ifndef mfakto_H_
#define mfakto_H_

//#define DETAILED_INFO
//#define CL_PERFORMANCE_INFO 
#define NUM_KERNELS 10
#define KERNEL_FILE "mfakto_Kernels.cl"
#define MAX_STREAMS 10

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
