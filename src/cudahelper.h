////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(int err, const char *file, const int line )
{
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line )
{
}

// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
    return 1;
}

void AllocateHostMemory(bool bPinGenericMemory, unsigned int **pp_a, unsigned int **ppAligned_a, int nbytes)
{
  // allocate host memory (pinned is required for achieve asynchronicity)
}

void FreeHostMemory(bool bPinGenericMemory, unsigned int **pp_a, unsigned int **ppAligned_a, int nbytes)
{
}

// end of CUDA Helper Functions
////////////////////////////////////////////////////////////////////////////////
