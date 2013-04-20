/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2013  Oliver Weihe (o.weihe@t-online.de)
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
*/
/* OpenCL specific code for trial factoring */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include "string.h"
#include "CL/cl.h"
#include "params.h"
#include "my_types.h"
#include "compatibility.h"
#include "read_config.h"
#include "parse.h"
#include "sieve.h"
#include "timer.h"
#include "checkpoint.h"
#include "filelocking.h"
#include "perftest.h"
#include "mfakto.h"
#include "output.h"
#include "gpusieve.h"
#ifndef _MSC_VER
#include <sys/time.h>
#else
#include "time.h"
#define time _time64
#define localtime _localtime64
#endif

// valgrind tests complain a lot about the blocks being uninitialized
#define malloc(x) calloc(x,1)

/* Global variables */

cl_uint             new_class=1;
cl_device_id        *devices;
cl_program          program;

cl_context          context=NULL;
cl_command_queue    commandQueue, commandQueuePrf=NULL;

#ifdef __cplusplus
extern "C"
{
#endif

#include "signal_handler.h"
extern mystuff_t    mystuff;
OpenCL_deviceinfo_t deviceinfo={{0}};
kernel_info_t       kernel_info[] = {
  /*   kernel (in sequence) | kernel function name | bit_min | bit_max | stages? | loaded kernel pointer */
     {   AUTOSELECT_KERNEL,   "auto",                  0,      0,         0,      NULL},
     {   _TEST_MOD_,          "test_k",                0,      0,         0,      NULL}, // used for various tests
     {   _71BIT_MUL24,        "mfakto_cl_71",         61,     71,         1,      NULL},
     {   _63BIT_MUL24,        "mfakto_cl_63",         58,     64,         1,      NULL},
     {   BARRETT70_MUL24,     "cl_barrett24_70",      64,     70,         0,      NULL},
     {   BARRETT79_MUL32,     "cl_barrett32_79",      64,     79,         1,      NULL},
     {   BARRETT77_MUL32,     "cl_barrett32_77",      64,     77,         1,      NULL},
     {   BARRETT76_MUL32,     "cl_barrett32_76",      64,     76,         1,      NULL},
     {   BARRETT92_MUL32,     "cl_barrett32_92",      65,     92,         0,      NULL},
     {   BARRETT88_MUL32,     "cl_barrett32_88",      65,     88,         0,      NULL},
     {   BARRETT87_MUL32,     "cl_barrett32_87",      65,     87,         0,      NULL},
     {   BARRETT73_MUL15,     "cl_barrett15_73",      60,     73,         0,      NULL},
     {   BARRETT69_MUL15,     "cl_barrett15_69",      60,     69,         0,      NULL},
     {   BARRETT70_MUL15,     "cl_barrett15_70",      60,     70,         0,      NULL},
     {   BARRETT71_MUL15,     "cl_barrett15_71",      60,     71,         0,      NULL},
     {   BARRETT88_MUL15,     "cl_barrett15_88",      60,     88,         0,      NULL},
     {   BARRETT83_MUL15,     "cl_barrett15_83",      60,     83,         0,      NULL},
     {   BARRETT82_MUL15,     "cl_barrett15_82",      60,     82,         0,      NULL},
     {   MG62,                "cl_mg62",              58,     62,         1,      NULL},
     {   MG88,                "cl_mg88",              58,     87,         1,      NULL},
     {   UNKNOWN_KERNEL,      "UNKNOWN kernel",        0,      0,         0,      NULL}, // end of automatic loading
     {   _64BIT_64_OpenCL,    "mfakto_cl_64",          0,     64,         0,      NULL}, // slow shift-cmp-sub kernel: removed
     {   BARRETT92_64_OpenCL, "cl_barrett32_92",      64,     92,         0,      NULL}, // mapped to 32-bit barrett so far
     {   CL_CALC_BIT_TO_CLEAR, "CalcBitToClear",       0,      0,         0,      NULL}, // called by gpusieve_init_class
     {   CL_CALC_MOD_INV,     "CalcModularInverses",   0,      0,         0,      NULL}, // called by gpusieve_init_exponent
     {   CL_SIEVE,            "SegSieve",              0,      0,         0,      NULL}, // GPU sieve
     {   BARRETT79_MUL32_GS,  "cl_barrett32_79_gs",   64,     10,         1,      NULL}, // keep the GPU-sieve-based kernels in the same order as their CPU-siev versions
     {   BARRETT77_MUL32_GS,  "cl_barrett32_77_gs",   64,     77,         1,      NULL},
     {   BARRETT76_MUL32_GS,  "cl_barrett32_76_gs",   64,     10,         1,      NULL},
     {   BARRETT92_MUL32_GS,  "cl_barrett32_92_gs",   65,     10,         0,      NULL},
     {   BARRETT88_MUL32_GS,  "cl_barrett32_88_gs",   65,     10,         0,      NULL},
     {   BARRETT87_MUL32_GS,  "cl_barrett32_87_gs",   65,     10,         0,      NULL},
     {   BARRETT73_MUL15_GS,  "cl_barrett15_73_gs",   60,     10,         0,      NULL},
     {   BARRETT69_MUL15_GS,  "cl_barrett15_69_gs",   60,     10,         0,      NULL},
     {   BARRETT70_MUL15_GS,  "cl_barrett15_70_gs",   60,     10,         0,      NULL},
     {   BARRETT71_MUL15_GS,  "cl_barrett15_71_gs",   60,     10,         0,      NULL},
     {   BARRETT88_MUL15_GS,  "cl_barrett15_88_gs",   60,     10,         0,      NULL},
     {   BARRETT83_MUL15_GS,  "cl_barrett15_83_gs",   60,     10,         0,      NULL},
     {   BARRETT82_MUL15_GS,  "cl_barrett15_82_gs",   60,     10,         0,      NULL},
};

/* save-save
     {   BARRETT79_MUL32_GS,  "cl_barrett32_79_gs",   64,     79,         1,      NULL}, // one kernel for all vector sizes
     {   BARRETT77_MUL32_GS,  "cl_barrett32_77_gs",   64,     77,         1,      NULL}, // one kernel for all vector sizes
     {   BARRETT76_MUL32_GS,  "cl_barrett32_76_gs",   64,     76,         1,      NULL}, // one kernel for all vector sizes
     {   BARRETT92_MUL32_GS,  "cl_barrett32_92_gs",   65,     92,         0,      NULL}, // one kernel for all vector sizes
     {   BARRETT88_MUL32_GS,  "cl_barrett32_88_gs",   65,     88,         0,      NULL}, // one kernel for all vector sizes
     {   BARRETT87_MUL32_GS,  "cl_barrett32_87_gs",   65,     87,         0,      NULL}, // one kernel for all vector sizes
     {   BARRETT73_MUL15_GS,  "cl_barrett15_73_gs",   60,     73,         0,      NULL}, // one kernel for all vector sizes
     {   BARRETT69_MUL15_GS,  "cl_barrett15_69_gs",   60,     69,         0,      NULL}, // one kernel for all vector sizes
     {   BARRETT70_MUL15_GS,  "cl_barrett15_70_gs",   60,     70,         0,      NULL}, // one kernel for all vector sizes
     {   BARRETT71_MUL15_GS,  "cl_barrett15_71_gs",   60,     71,         0,      NULL}, // one kernel for all vector sizes
     {   BARRETT88_MUL15_GS,  "cl_barrett15_88_gs",   60,     88,         0,      NULL}, // one kernel for all vector sizes
     {   BARRETT83_MUL15_GS,  "cl_barrett15_83_gs",   60,     83,         0,      NULL}, // one kernel for all vector sizes
     {   BARRETT82_MUL15_GS,  "cl_barrett15_82_gs",   60,     82,         0,      NULL}, // one kernel for all vector sizes
*/
void printArray(const char * Name, const cl_uint * Data, const cl_uint len, cl_uint hex=0)
{
  cl_uint i, o, c, val;
  char *fmt1, *fmt2, *fmt3, *fmt4;

  if (hex)
  {
    fmt1="<%u x %#x> ";
    fmt2="%#x ";
    fmt3="... %#x %#x %#x\n";
    fmt4="<%d x 0x0 at the end>\n";
  }
  else
  {
    fmt1="<%u x %u> ";
    fmt2="%u ";
    fmt3="... %u %u %u\n";
    fmt4="<%d x 0 at the end>\n";
  }
  o = printf("%s (%d): ", Name, len);
  for(i = 0; i < len-2 && o < 960;) // no more than 1000 chars
  {
    if (Data[i] == Data[i+1] && Data[i] == Data[i+2])
    {
      val = Data[i];
      c = 0;
      while(Data[i] == val && i < len)
      {
        ++c; ++i;
      }
      o += printf(fmt1, c, val);
      continue;
    }
    else
    {
      o += printf(fmt2, Data[i]);
    }
    ++i;
  }
  if (i<len) printf(fmt3, Data[len-3], Data[len-2], Data[len-1]); else printf("\n");
  i=len-1; c=0;
  while ((Data[i--] == 0) && i>0) c++;
  if (c > 0) printf(fmt4, c);
}

/* allocate memory buffer arrays, test a small kernel */
int init_CLstreams(void)
{
  cl_uint i;
  cl_int status;

  if (context==NULL)
  {
    fprintf(stderr, "invalid context.\n");
    return 1;
  }

  for(i=0;i<(mystuff.num_streams);i++)
  {
    mystuff.stream_status[i] = UNUSED;
    if( (mystuff.h_ktab[i] = (cl_uint *) malloc( mystuff.threads_per_grid * sizeof(cl_uint))) == NULL )
    {
      printf("ERROR: malloc(h_ktab[%d]) failed\n", i);
      return 1;
    }
    mystuff.d_ktab[i] = clCreateBuffer(context, 
                      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                      mystuff.threads_per_grid * sizeof(cl_uint),
                      mystuff.h_ktab[i], 
                      &status);
    if(status != CL_SUCCESS) 
  	{ 
	  	std::cout<<"Error " << status << ": clCreateBuffer (h_ktab[" << i << "]) \n";
	  	return 1;
	  }
  }
  if( (mystuff.h_RES = (cl_uint *) malloc(32 * sizeof(cl_uint))) == NULL )
  {
    printf("ERROR: malloc(h_RES) failed\n");
    return 1;
  }
  mystuff.d_RES = clCreateBuffer(context, 
                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    32 * sizeof(cl_uint),
                    mystuff.h_RES, 
                    &status);
  if(status != CL_SUCCESS) 
  { 
		std::cout<<"Error " << status << ": clCreateBuffer (d_RES)\n";
		return 1;
	}
#ifdef CHECKS_MODBASECASE
  if( (mystuff.h_modbasecase_debug = (cl_uint *) malloc(32 * sizeof(cl_uint))) == NULL )
  {
    printf("ERROR: malloc(h_modbasecase_debug) failed\n");
    return 1;
  }
  mystuff.d_modbasecase_debug = clCreateBuffer(context,
                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    32 * sizeof(cl_uint),
                    mystuff.h_modbasecase_debug,
                    &status);
  if(status != CL_SUCCESS) 
  {
		std::cout<<"Error " << status << ": clCreateBuffer (d_modbasecase_debug)\n";
		return 1;
	}
#endif

  if (mystuff.gpu_sieving == 1)
  {
    // alloc GPU buffers, calculate the prime info and copy to device
    gpusieve_init(&mystuff, context);

    // now already set the fix parameters for the GPU sieve kernels
    // CL_CALC_MOD_INV
    // CalcModularInverses<<<primes_per_thread+1, threadsPerBlock>>>(mystuff->exponent, (int *)mystuff->d_calc_bit_to_clear_info);

    status = clSetKernelArg(kernel_info[CL_CALC_MOD_INV].kernel,
                      1,
                      sizeof(cl_mem),
                      (void *)&mystuff.d_calc_bit_to_clear_info);
    if(status != CL_SUCCESS)
	  {
		  std::cout<<"Error " << status << ": Setting kernel argument. (d_calc_bit_to_clear_info)\n";
		  return 1;
	  }

    // CL_CALC_BIT_TO_CLEAR
	  // CalcBitToClear<<<primes_per_thread+1, threadsPerBlock>>>(mystuff->exponent, k_base, (int *)mystuff->d_calc_bit_to_clear_info, (cl_uchar *)mystuff->d_sieve_info);

    status = clSetKernelArg(kernel_info[CL_CALC_BIT_TO_CLEAR].kernel,
                      2,
                      sizeof(cl_mem),
                      (void *)&mystuff.d_calc_bit_to_clear_info);
    if(status != CL_SUCCESS)
	  {
		  std::cout<<"Error " << status << ": Setting kernel argument. (d_calc_bit_to_clear_info)\n";
		  return 1;
	  }
    status = clSetKernelArg(kernel_info[CL_CALC_BIT_TO_CLEAR].kernel,
                      3,
                      sizeof(cl_mem),
                      (void *)&mystuff.d_sieve_info);
    if(status != CL_SUCCESS)
	  {
		  std::cout<<"Error " << status << ": Setting kernel argument. (d_sieve_info)\n";
		  return 1;
	  }

    // CL_SIEVE
    // SegSieve<<<(sieve_size + block_size - 1) / block_size, threadsPerBlock>>>((cl_uchar *)mystuff->d_bitarray, (cl_uchar *)mystuff->d_sieve_info, primes_per_thread);
    status = clSetKernelArg(kernel_info[CL_SIEVE].kernel,
                      0,
                      sizeof(cl_mem),
                      (void *)&mystuff.d_bitarray);
    if(status != CL_SUCCESS)
	  {
		  std::cout<<"Error " << status << ": Setting kernel argument. (d_bitarray)\n";
		  return 1;
	  }
    status = clSetKernelArg(kernel_info[CL_SIEVE].kernel,
                      1,
                      sizeof(cl_mem),
                      (void *)&mystuff.d_sieve_info);
    if(status != CL_SUCCESS)
	  {
		  std::cout<<"Error " << status << ": Setting kernel argument. (d_sieve_info)\n";
		  return 1;
	  }
    // param 2 (primes_per_thread) is variable, can't set it now.
  }

	return 0;
}


/*
 * init_CL: all OpenCL-related one-time inits:
 *   create context, devicelist, command queue, 
 *   load kernel file, compile, link CL source, build program and kernels
 */
int init_CL(int num_streams, cl_int devnumber)
{
  cl_int status;
  size_t dev_s;
  cl_uint numplatforms, i;
  cl_platform_id platform = NULL;
  cl_platform_id* platformlist = NULL;
  cl_device_type devtype = CL_DEVICE_TYPE_GPU;

  printf("Select device - "); fflush(NULL);
  status = clGetPlatformIDs(0, NULL, &numplatforms);
  if(status != CL_SUCCESS)
  {
    std::cerr << "Error " << status << ": clGetPlatformsIDs(num)\n";
    return 1;
  }

  if (devnumber < 0)
  {
    devtype = CL_DEVICE_TYPE_CPU;
    devnumber = 0;
    printf("(CPU) - "); fflush(NULL);
  }

  if (numplatforms > 0)
  {
    platformlist = new cl_platform_id[numplatforms];
    status = clGetPlatformIDs(numplatforms, platformlist, NULL);
    if(status != CL_SUCCESS)
    {
      std::cerr << "Error " << status << ": clGetPlatformsIDs\n";
      return 1;
    }

    if (devnumber > 10) // platform number specified as part of -d
    {
      i = devnumber/10 - 1;
      if (i < numplatforms)
      {
        platform = platformlist[i];
#ifdef DETAILED_INFO
        char buf[128];
        status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR,
                        sizeof(buf), buf, NULL);
        if(status != CL_SUCCESS)
        {
          std::cerr << "Error " << status << ": clGetPlatformInfo(VENDOR)\n";
          return 1;
        }
        std::cout << "OpenCL Platform " << i+1 << "/" << numplatforms << ": " << buf;

        status = clGetPlatformInfo(platform, CL_PLATFORM_VERSION,
                        sizeof(buf), buf, NULL);
        if(status != CL_SUCCESS)
        {
          std::cerr << "Error " << status << ": clGetPlatformInfo(VERSION)\n";
          return 1;
        }
        std::cout << ", Version: " << buf << std::endl;
#endif
      }
      else
      {
        fprintf(stderr, "Error: Only %d platforms found. Cannot use platform %d (bad parameter to option -d).\n", numplatforms, i+1);
        return 1;
      }
    }
    else for(i=0; i < numplatforms; i++) // autoselect: search for AMD
    {
      char buf[128];
      status = clGetPlatformInfo(platformlist[i], CL_PLATFORM_VENDOR,
                        sizeof(buf), buf, NULL);
      if(status != CL_SUCCESS)
      {
        std::cerr << "Error " << status << ": clGetPlatformInfo(VENDOR)\n";
        return 1;
      }
      if(strcmp(buf, "Advanced Micro Devices, Inc.") == 0)
      {
        platform = platformlist[i];
      }
#ifdef DETAILED_INFO
      std::cout << "OpenCL Platform " << i+1 << "/" << numplatforms << ": " << buf;

      status = clGetPlatformInfo(platformlist[i], CL_PLATFORM_VERSION,
                        sizeof(buf), buf, NULL);
      if(status != CL_SUCCESS)
      {
        std::cerr << "Error " << status << ": clGetPlatformInfo(VERSION)\n";
        return 1;
      }
      std::cout << ", Version: " << buf << std::endl;
#endif
    }
  }

  delete[] platformlist;
  
  if(platform == NULL)
  {
    std::cerr << "Error: No platform found\n";
    return 1;
  }

  cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
  context = clCreateContextFromType(cps, devtype, NULL, NULL, &status);
  if (status == CL_DEVICE_NOT_FOUND)
  {
    clReleaseContext(context);
    std::cout << "GPU not found, fallback to CPU." << std::endl;
    context = clCreateContextFromType(cps, CL_DEVICE_TYPE_CPU, NULL, NULL, &status);
    if(status != CL_SUCCESS) 
  	{  
   	  std::cerr << "Error " << status << ": clCreateContextFromType(CPU)\n";
	    return 1; 
    }
  }
  else if(status != CL_SUCCESS) 
	{  
		std::cerr << "Error " << status << ": clCreateContextFromType(GPU)\n";
		return 1;
  }

  cl_uint num_devices;
  status = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(num_devices), &num_devices, NULL);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr << "Error " << status << ": clGetContextInfo(CL_CONTEXT_NUM_DEVICES) - assuming one device\n";
		// return 1;
    num_devices = 1;
	}

  status = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &dev_s);
  if(status != CL_SUCCESS) 
	{  
		std::cerr << "Error " << status << ": clGetContextInfo(numdevs)\n";
		return 1;
	}

	if(dev_s == 0)
	{
		std::cerr << "Error: no devices.\n";
		return 1;
	}

  devices = (cl_device_id *)malloc(dev_s*sizeof(cl_device_id));  // *sizeof(...) should not be needed (dev_s is in bytes)
	if(devices == 0)
	{
		std::cerr << "Error: Out of memory.\n";
		return 1;
	}

  status = clGetContextInfo(context, CL_CONTEXT_DEVICES, dev_s*sizeof(cl_device_id), devices, NULL);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr << "Error " << status << ": clGetContextInfo(devices)\n";
		return 1;
	}

  devnumber = devnumber % 10;  // use only the last digit as device number, counting from 1
  cl_uint dev_from=0, dev_to=num_devices;
  if (devnumber > 0)
  {
    if ((cl_uint)devnumber > num_devices)
    {
      fprintf(stderr, "Error: Only %d devices found. Cannot use device %d (bad parameter to option -d).\n", num_devices, devnumber);
      return 1;
    }
    else
    {
      dev_to    = devnumber;    // tweak the loop to run only once for our device
      dev_from  = --devnumber;  // index from 0
    }
  }
  printf("Get device info - "); fflush(stdout);
   
  for (i=dev_from; i<dev_to; i++)
  {
    status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(deviceinfo.d_name), deviceinfo.d_name, NULL);
    if(status != CL_SUCCESS)
	  {
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_NAME)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(deviceinfo.d_ver), deviceinfo.d_ver, NULL);
    if(status != CL_SUCCESS)
	  {
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_VERSION)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(deviceinfo.v_name), deviceinfo.v_name, NULL);
    if(status != CL_SUCCESS)
	  {
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_VENDOR)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(deviceinfo.dr_version), deviceinfo.dr_version, NULL);
    if(status != CL_SUCCESS)
	  {
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DRIVER_VERSION)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(deviceinfo.exts), deviceinfo.exts, NULL);
    if(status != CL_SUCCESS)
	  {
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_EXTENSIONS)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(deviceinfo.gl_cache), &deviceinfo.gl_cache, NULL);
    if(status != CL_SUCCESS)
	  {
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceinfo.gl_mem), &deviceinfo.gl_mem, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_GLOBAL_MEM_SIZE)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(deviceinfo.max_clock), &deviceinfo.max_clock, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(deviceinfo.units), &deviceinfo.units, NULL);
    if(status != CL_SUCCESS)
	  {
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_MAX_COMPUTE_UNITS)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(deviceinfo.wg_size), &deviceinfo.wg_size, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(deviceinfo.w_dim), &deviceinfo.w_dim, NULL);
    if(status != CL_SUCCESS)
	  {
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(deviceinfo.wi_sizes), deviceinfo.wi_sizes, NULL);
    if(status != CL_SUCCESS)
	  {
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES)\n";
	  	return 1;
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(deviceinfo.l_mem), &deviceinfo.l_mem, NULL);
    if(status != CL_SUCCESS)
	  {
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_LOCAL_MEM_SIZE)\n";
	  	return 1;
	  }
    
#ifdef DETAILED_INFO
    std::cout << "Device " << i+1  << "/" << num_devices << ": " << deviceinfo.d_name << " (" << deviceinfo.v_name << "),\ndevice version: "
      << deviceinfo.d_ver << ", driver version: " << deviceinfo.dr_version << "\nExtensions: " << deviceinfo.exts
      << "\nGlobal memory:" << deviceinfo.gl_mem << ", Global memory cache: " << deviceinfo.gl_cache
      << ", local memory: " << deviceinfo.l_mem << ", workgroup size: " << deviceinfo.wg_size << ", Work dimensions: " << deviceinfo.w_dim
      << "[" << deviceinfo.wi_sizes[0] << ", " << deviceinfo.wi_sizes[1] << ", " << deviceinfo.wi_sizes[2] << ", " << deviceinfo.wi_sizes[3] << ", " << deviceinfo.wi_sizes[4]
      << "] , Max clock speed:" << deviceinfo.max_clock << ", compute units:" << deviceinfo.units << std::endl;
#endif // DETAILED_INFO
  }

  if (strstr(deviceinfo.exts, "global_int32_base_atomics") == NULL)
  {
    printf("\nWARNING: Device does not support atomic operations. This may lead to errors\n"
           "         when multiple factors are found in the same block. Possible errors\n"
           "         include reporting just one of the factors, or (less likely) scrambled\n"
           "         factors. If the reported factor(s) are not accepted by primenet,\n"
           "         please re-run this test on the CPU, or on a GPU with atomics.\n");
  }

  deviceinfo.maxThreadsPerBlock = deviceinfo.wi_sizes[0];
  deviceinfo.maxThreadsPerGrid  = deviceinfo.wi_sizes[0];
  for (i=1; i<deviceinfo.w_dim && i<5; i++)
  {
    if (deviceinfo.wi_sizes[i])
      deviceinfo.maxThreadsPerGrid *= deviceinfo.wi_sizes[i];
  }

//  cl_command_queue_properties props = 0;  // serialize data transfer and kernel execution (i.e. no overlapping transfer while some kernel is still running)
  cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;  // kernels and copy-jobs are queued with event dependencies, so this should work ...
                                                                               // but so far the GPU driver does not support that anyway (as of Catalyst 12.9)

  commandQueue = clCreateCommandQueue(context, devices[devnumber], props, &status);
  if(status != CL_SUCCESS)
	{
    std::cerr << "Error " << status << ": clCreateCommandQueue(dev#" << devnumber+1 << ")\n";
		return 1;
	}

  props |= CL_QUEUE_PROFILING_ENABLE;

  commandQueuePrf = clCreateCommandQueue(context, devices[devnumber], props, &status);
  if(status != CL_SUCCESS)
	{
    std::cerr << "Error " << status << ": clCreateCommandQueuePrf(dev#" << devnumber+1 << ")\n";
		return 1;
	}

 	size_t size;
	char*  source;

	std::fstream f(KERNEL_FILE, (std::fstream::in | std::fstream::binary));

	if(f.is_open())
	{
		f.seekg(0, std::fstream::end);
		size = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);

		source = (char *) malloc(size+1);
		if(!source)
		{
			f.close();
      std::cerr << "\noom\n";
			return 1;
		}

		f.read(source, size);
		f.close();
		source[size] = '\0';
  }
	else
	{
		std::cerr << "\nKernel file \""KERNEL_FILE"\" not found, it needs to be in the same directory as the executable.\n";
		return 1;
	}

  program = clCreateProgramWithSource(context, 1, (const char **)&source, &size, &status);
	if(status != CL_SUCCESS)
	{
	  std::cerr << "Error " << status << ": clCreateProgramWithSource\n";
	  return 1;
	}

  char program_options[150];
  // so far use the same vector size for all kernels ...
  sprintf(program_options, "-I. -DVECTOR_SIZE=%d", mystuff.vectorsize);
#ifdef CL_DEBUG
  strcat(program_options, " -g");
#else
  if (mystuff.gpu_type != GPU_NVIDIA) strcat(program_options, " -O3");
#endif

#ifdef MORE_CLASSES
  strcat(program_options, " -DMORE_CLASSES");
#endif

#ifdef CHECKS_MODBASECASE
  strcat(program_options, " -DCHECKS_MODBASECASE");
#endif

  if (mystuff.gpu_sieving == 1)
    strcat(program_options, " -DCL_GPU_SIEVE");

  if (mystuff.small_exp == 1)
    strcat(program_options, " -DSMALL_EXP");

  if (mystuff.CompileOptions[0])  // if mfakto.ini defined compile options, override the default with them
    strcpy(program_options, mystuff.CompileOptions);

#ifdef DETAILED_INFO
  printf("Compiling kernels (build options: \"%s\").", program_options);
#else
  printf("Compiling kernels .");
#endif
  fflush(stdout);
  // program_options can be overridden by setting en environment variable AMD_OCL_BUILD_OPTIONS

  status = clBuildProgram(program, 1, &devices[devnumber], program_options, NULL, NULL);
  if(status != CL_SUCCESS)
  {
    if(status == CL_BUILD_PROGRAM_FAILURE)
    {
      cl_int logstatus;
      char *buildLog = NULL;
      size_t buildLogSize = 0;
      logstatus = clGetProgramBuildInfo (program, devices[devnumber], CL_PROGRAM_BUILD_LOG, 
                buildLogSize, buildLog, &buildLogSize);
      if(logstatus != CL_SUCCESS)
      {
        std::cerr << "Error " << logstatus << ": clGetProgramBuildInfo failed.";
        return 1;
      }
      buildLog = (char*)calloc(buildLogSize,1);
      if(buildLog == NULL)
      {
        std::cerr << "\noom\n";
        return 1;
      }
      fflush(NULL);
      logstatus = clGetProgramBuildInfo (program, devices[devnumber], CL_PROGRAM_BUILD_LOG, 
                buildLogSize, buildLog, NULL);
      if(logstatus != CL_SUCCESS)
      {
        std::cerr << "Error " << logstatus << ": clGetProgramBuildInfo failed.";
        free(buildLog);
        return 1;
      }

      std::cout << " \n\tBUILD OUTPUT\n";
      std::cout << buildLog << std::endl;
      std::cout << " \tEND OF BUILD OUTPUT\n";
      free(buildLog);
    }
		std::cerr<<"Error " << status << ": clBuildProgram\n";
		return 1; 
  }

  free(source);
  /* get kernels by name */
  if (mystuff.gpu_sieving == 0)
  {
    for (i=_TEST_MOD_; i<UNKNOWN_KERNEL; i++)
    {
      putchar('.'); fflush(stdout);
      kernel_info[i].kernel = clCreateKernel(program, kernel_info[i].kernelname, &status);
      if(status != CL_SUCCESS)
      {
        std::cerr<<"Error " << status << ": Creating Kernel " << kernel_info[i].kernelname << " from program. (clCreateKernel)\n";
        return 1;
      }
    }
  }
  else
  {
    for (i=CL_CALC_BIT_TO_CLEAR; i<=BARRETT82_MUL15_GS; i++)
    {
      putchar('.'); fflush(stdout);
      kernel_info[i].kernel = clCreateKernel(program, kernel_info[i].kernelname, &status);
      if(status != CL_SUCCESS)
      {
        std::cerr<<"Error " << status << ": Creating Kernel " << kernel_info[i].kernelname << " from program. (clCreateKernel)\n";
        return 1;
      }
    }
  }

  printf("\n");
  return 0;
}


/* Run the seg_sieve kernel to do the FC sieving
   numblocks and localThreads: correspond to cuda's numblocks and threadsPerBlock,
   device_big_bit_array:  the sieve array, resides only on the GPU (except for debugging)
   device_pinfo:          a block of info of the primes used for sieving
   copy_event:            is used to synchronize the previous copy-to-device of pinfo with
                          this kernel.
   pinfo_size:            for statistics, used only when running with CL_PERFORMANCE_INFO defined
*/
cl_int run_seg_sieve(cl_uint numblocks, cl_uint localThreads,
                   cl_mem device_big_bit_array, cl_mem device_pinfo,
                   cl_uint primes_per_thread,
                   cl_event copy_event, cl_uint pinfo_size)
{
  cl_int status;
  cl_event run_event;
  static cl_uint first_run=1;
  size_t globalThreads = numblocks * localThreads;


  if (first_run)
  {
    // set all the kernel parameters that don't change
    first_run=0;
    status = clSetKernelArg(kernel_info[CL_SIEVE].kernel,
                    0,
                    sizeof(cl_mem),
                    (void *)&device_big_bit_array);
    if(status != CL_SUCCESS)
	  {
		  std::cerr<<"Error " << status << ": Setting kernel argument. (device_big_bit_array)\n";
		  return 1;
	  }
    status = clSetKernelArg(kernel_info[CL_SIEVE].kernel,
                    1,
                    sizeof(cl_mem),
                    (void *)&device_pinfo);
    if(status != CL_SUCCESS)
	  {
		  std::cerr<<"Error " << status << ": Setting kernel argument. (device_pinfo)\n";
		  return 1;
	  }
    status = clSetKernelArg(kernel_info[CL_SIEVE].kernel,
                    2,
                    sizeof(cl_uint),
                    (void *)&primes_per_thread);
    if(status != CL_SUCCESS)
	  {
		  std::cerr<<"Error " << status << ": Setting kernel argument. (primes_per_thread)\n";
		  return 1;
	  }
  }
  status = clEnqueueNDRangeKernel(QUEUE,
                 kernel_info[CL_SIEVE].kernel,
                 1,
                 NULL,
                 &globalThreads,
                 (size_t *)&localThreads,
                 1,
                 &copy_event, // wait for the primes write to finish
                 &run_event);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<< "Error " << status << ": Enqueuing kernel(clEnqueueNDRangeKernel)" << "\n";
		return 1;
	}
  clFinish(QUEUE);
#ifdef CL_PERFORMANCE_INFO
              cl_ulong startTime=0;
              cl_ulong endTime=1000;
              /* Get kernel profiling info */
              status = clGetEventProfilingInfo(copy_event,
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &startTime,
                                0);
              if(status != CL_SUCCESS)
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
                return RET_ERROR;
              }
              status = clGetEventProfilingInfo(copy_event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endTime,
                                0);
              if(status != CL_SUCCESS) 
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
                return RET_ERROR;
              }
              printf("primes array copied in %2.2f ms (%4.2f MB/s), ", (endTime - startTime)/1e6,
                       pinfo_size * 1e3 / (endTime - startTime) );
              status = clGetEventProfilingInfo(run_event,
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &startTime,
                                0);
              if(status != CL_SUCCESS)
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
                return RET_ERROR;
              }
              status = clGetEventProfilingInfo(run_event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endTime,
                                0);
              if(status != CL_SUCCESS)
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
                return RET_ERROR;
              }
              printf("proc'd in %2.2f ms (%3.2f M/s)\n", (endTime - startTime)/1e6, double(globalThreads) *1e3/ (endTime - startTime));
#endif
              status = clReleaseEvent(copy_event);
              if(status != CL_SUCCESS) 
              { 
		         	  std::cerr<< "Error " << status << ": Release copy event object. (clReleaseEvent)\n";
		         	  return RET_ERROR;
    	       	}
              status = clReleaseEvent(run_event);
             	if(status != CL_SUCCESS) 
           	  { 
		          	std::cerr<< "Error " << status << ": Release run event object. (clReleaseEvent)\n";
		         	  return RET_ERROR;
            	}

  return status;
}

#ifdef __cplusplus
}
#endif


/* error callback function - not used right now */
void  CL_CALLBACK CL_error_cb(const char *errinfo,
  	const void  *private_info,
  	size_t  cb,
  	void  *user_data)
{
  std::cerr << "Error callback: " << errinfo << std::endl;
}

/* Run the CalcModularInverses kernel
__kernel void __attribute__((work_group_size_hint(256, 1, 1))) CalcModularInverses (uint exponent, __global int *calc_info)

   numblocks and localThreads: correspond to cuda's numblocks and threadsPerBlock,
   run_event:             can be used to synchronize the following copy-to-host of the
                          block_counts. But for now the kernel is run synchronously, so this
                          is not needed.
*/
cl_int run_calc_mod_inv(cl_uint numblocks, size_t localThreads, cl_event *run_event)
{
  cl_int status;
  size_t globalThreads = numblocks * localThreads;

#ifdef DETAILED_INFO
    printf("run_calc_mod_inv: %d x %d = %d threads, exp=%u\n",
        (int) numblocks, (int) localThreads, (int) globalThreads, mystuff.exponent);
#endif

  status = clSetKernelArg(kernel_info[CL_CALC_MOD_INV].kernel,
                    0,
                    sizeof(cl_uint),
                    (void *)&mystuff.exponent);
  if(status != CL_SUCCESS)
	{
	  std::cout<<"Error " << status << ": Setting kernel argument. (exponent)\n";
	  return 1;
	}

#ifdef CL_PERFORMANCE_INFO
  if (run_event == NULL) run_event = &mystuff.copy_events[0]; // When checking performance, we need an event to monitor.
#endif

  status = clEnqueueNDRangeKernel(QUEUE,
                 kernel_info[CL_CALC_MOD_INV].kernel,
                 1,
                 NULL,
                 &globalThreads,
                 &localThreads,
                 0,
                 NULL,
                 run_event);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<< "Error " << status << ": Enqueuing kernel(clEnqueueNDRangeKernel)" << "\n";
		return 1;
	}
#ifdef CL_PERFORMANCE_INFO
  clFinish(QUEUE);
              cl_ulong startTime=0;
              cl_ulong endTime=1000;
              /* Get kernel profiling info */
              status = clGetEventProfilingInfo(*run_event,
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &startTime,
                                0);
              if(status != CL_SUCCESS)
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
                return RET_ERROR;
              }
              status = clGetEventProfilingInfo(*run_event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endTime,
                                0);
              if(status != CL_SUCCESS)
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
                return RET_ERROR;
              }
              printf("CalcModularInverses in %2.2f us (%3.2f M/s)\n", (endTime - startTime)/1e3, double(globalThreads) *1e3/ (endTime - startTime));
              clReleaseEvent(mystuff.copy_events[0]); // dont use run_events - if it was passed in, it will still be needed. Ignore errors here, as we may have used a different event
#endif

#ifdef DETAILED_INFO
    // get mystuff.d_calc_bit_to_clear_info and print it
	cl_uint rowinfo_size = MAX_PRIMES_PER_THREAD*4 * sizeof (cl_uint) + mystuff.gpu_sieve_primes * 8;

  status = clEnqueueReadBuffer(QUEUE,     // only for tracing/verification - not needed later.
                mystuff.d_calc_bit_to_clear_info,
                CL_TRUE,
                0,
                rowinfo_size,
                mystuff.h_calc_bit_to_clear_info,
                0,
                NULL,
                NULL);
    
  if(status != CL_SUCCESS) 
	{ 
    std::cout << "Error " << status << ": clEnqueueReadBuffer d_calc_bit_to_clear_info failed. (clEnqueueReadBuffer)\n";
		return 1;
  }

  printArray("h_calc_bit_to_clear_info", mystuff.h_calc_bit_to_clear_info, rowinfo_size/sizeof(int), 1);
#endif

  return status;
}

/* Run the CalcBitToClear kernel
__kernel void __attribute__((work_group_size_hint(256, 1, 1))) CalcBitToClear (uint exponent, ulong k_base, __global int *calc_info, __global uchar *pinfo_dev)

   numblocks and localThreads: correspond to cuda's numblocks and threadsPerBlock,
   run_event:                  can be used to synchronize the following calls.
   k_min:                      starting k for the calculation (passed to the kernel as k_base)
*/
cl_int run_calc_bit_to_clear(cl_uint numblocks, size_t localThreads, cl_event *run_event, cl_ulong k_min)
{
  static int last_exponent = 0;
  cl_int   status;
  size_t   globalThreads = numblocks * localThreads;

#ifdef DETAILED_INFO
    printf("run_calc_bit_to_clear: %d x %d = %d threads, exp=%u, k_min=%llu\n",
        (int) numblocks, (int) localThreads, (int) globalThreads, mystuff.exponent, (long long unsigned int) k_min);
#endif

  if (last_exponent != mystuff.exponent) // only copy the exponent if it changed
  {
    last_exponent = mystuff.exponent;
    status = clSetKernelArg(kernel_info[CL_CALC_BIT_TO_CLEAR].kernel,
                    0,
                    sizeof(cl_uint),
                    (void *)&mystuff.exponent);
    if(status != CL_SUCCESS)
  	{
  		std::cout<<"Error " << status << ": Setting kernel argument. (exp)\n";
    	return 1;
    }
	}
  status = clSetKernelArg(kernel_info[CL_CALC_BIT_TO_CLEAR].kernel,
                    1,
                    sizeof(cl_ulong),
                    (void *)&k_min);
  if(status != CL_SUCCESS)
	{
		std::cout<<"Error " << status << ": Setting kernel argument. (k_min)\n";
		return 1;
	}

#ifdef CL_PERFORMANCE_INFO
  if (run_event == NULL) run_event = &mystuff.copy_events[0];  // When checking performance, we need an event to monitor.
#endif

  status = clEnqueueNDRangeKernel(QUEUE,
                 kernel_info[CL_CALC_BIT_TO_CLEAR].kernel,
                 1,
                 NULL,
                 &globalThreads,
                 &localThreads,
                 0,
                 NULL,
                 run_event);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<< "Error " << status << ": Enqueuing kernel(clEnqueueNDRangeKernel)\n";
		return 1;
	}
  clFlush(QUEUE);

#ifdef CL_PERFORMANCE_INFO
  clFinish(QUEUE);
  cl_ulong startTime=0;
  cl_ulong endTime=1000;
  /* Get kernel profiling info */
  status = clGetEventProfilingInfo(*run_event,
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &startTime,
                                0);
  if(status != CL_SUCCESS)
 	{ 
		std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
    return RET_ERROR;
  }
  status = clGetEventProfilingInfo(*run_event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endTime,
                                0);
  if(status != CL_SUCCESS) 
 	{ 
		std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
    return RET_ERROR;
  }
  std::cout<< "CalcBitToClear " << globalThreads << " primes: " << (endTime - startTime)/1e3 << " us ("
                       << globalThreads * 1e3 / (endTime - startTime) << " M/s)\n" ;
  clReleaseEvent(mystuff.copy_events[0]); // ignore errors: we may have use a different event
#endif

#ifdef DETAILED_INFO
    // get mystuff.d_calc_bit_to_clear_info and d_sieve_info and print it
	cl_uint info_size = MAX_PRIMES_PER_THREAD*4 * sizeof (cl_uint) + mystuff.gpu_sieve_primes * 8;
  status = clEnqueueReadBuffer(QUEUE,     // only for tracing/verification - not needed later.
                mystuff.d_calc_bit_to_clear_info,
                CL_TRUE,
                0,
                info_size,
                mystuff.h_calc_bit_to_clear_info,
                0,
                NULL,
                NULL);
    
  if(status != CL_SUCCESS) 
	{ 
    std::cout << "Error " << status << ": clEnqueueReadBuffer d_calc_bit_to_clear_info failed. (clEnqueueReadBuffer)\n";
		return 1;
  }

  printArray("h_calc_bit_to_clear_info", mystuff.h_calc_bit_to_clear_info, info_size/sizeof(int), 1);

  info_size = mystuff.sieve_size;

  status = clEnqueueReadBuffer(QUEUE,     // only for tracing/verification - not needed later.
                mystuff.d_sieve_info,
                CL_TRUE,
                0,
                info_size,
                mystuff.h_sieve_info,
                0,
                NULL,
                NULL);
    
  if(status != CL_SUCCESS) 
	{ 
    std::cout << "Error " << status << ": clEnqueueReadBuffer h_sieve_info failed. (clEnqueueReadBuffer)\n";
		return 1;
  }

  printArray("h_sieve_info", mystuff.h_sieve_info, info_size/sizeof(int), 1);
  //#endif
#endif

	return 0;
}

/* Run the SegSieve kernel
__kernel void __attribute__((work_group_size_hint(256, 1, 1))) SegSieve (__global uchar *big_bit_array_dev, __global uchar *pinfo_dev, uint maxp)

   numblocks and localThreads: correspond to cuda's numblocks and threadsPerBlock,
   run_event:                  can be used to synchronize the following calls.
   maxp:                       numer of primes per thread (passed to the kernel as maxp)
*/
cl_int run_cl_sieve(cl_uint numblocks, size_t localThreads, cl_event *run_event, cl_uint maxp)
{
  static cl_uint last_maxp = 0xFFFFFFFF;  // 0 is a bad choice for "uninitialized" as it can happen for small GPUSievePrimes
  cl_int         status;
  size_t         globalThreads = numblocks * localThreads;

#ifdef DETAILED_INFO
    printf("run_cl_sieve: %d x %d = %d threads, exp=%u, maxp=%d\n",
        (int) numblocks, (int) localThreads, (int) globalThreads, mystuff.exponent, maxp);
#endif

  if (last_maxp != maxp) // only copy primes-per-thread if it changed
  {
    last_maxp = maxp;
    status = clSetKernelArg(kernel_info[CL_SIEVE].kernel,
                    2,
                    sizeof(cl_uint),
                    (void *)&maxp);
    if(status != CL_SUCCESS)
  	{
  		std::cout<<"Error " << status << ": Setting kernel argument. (exp)\n";
    	return 1;
    }
	}

#ifdef CL_PERFORMANCE_INFO
  if (run_event == NULL) run_event = &mystuff.copy_events[0];  // When checking performance, we need an event to monitor.
#endif

  status = clEnqueueNDRangeKernel(QUEUE,
                 kernel_info[CL_SIEVE].kernel,
                 1,
                 NULL,
                 &globalThreads,
                 &localThreads,
                 0,
                 NULL,
                 run_event);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<< "Error " << status << ": Enqueuing kernel(clEnqueueNDRangeKernel)\n";
		return 1;
	}
  clFlush(QUEUE);
/////////////////////////////////////////////////
#ifdef CL_PERFORMANCE_INFO
  clFinish(QUEUE);
  cl_ulong startTime=0;  // device time in nanosecs
  cl_ulong endTime=1000;
  /* Get kernel profiling info */
  status = clGetEventProfilingInfo(*run_event,
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &startTime,
                                0);
  if(status != CL_SUCCESS)
 	{ 
		std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
    return RET_ERROR;
  }
  status = clGetEventProfilingInfo(*run_event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endTime,
                                0);
  if(status != CL_SUCCESS) 
 	{ 
		std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
    return RET_ERROR;
  }
  std::cout<< "sieve using " << globalThreads << " threads: " << (endTime - startTime)/1e6 << " ms ("
                       << globalThreads * 1e3 / (endTime - startTime) << " M/s), " <<
                       mystuff.gpu_sieve_size * 1e3 / (endTime - startTime) << " M FCs/s sieved\n";
  clReleaseEvent(mystuff.copy_events[0]);
#endif

#ifdef DETAILED_INFO
  //mystuff->d_bitarray, (cl_uchar *)mystuff->d_sieve_info
  cl_uint info_size = mystuff.sieve_size;

  status = clEnqueueReadBuffer(QUEUE,     // only for tracing/verification - not needed later.
                mystuff.d_sieve_info,
                CL_TRUE,
                0,
                info_size,
                mystuff.h_sieve_info,
                0,
                NULL,
                NULL);
    
  if(status != CL_SUCCESS) 
	{ 
    std::cout << "Error " << status << ": clEnqueueReadBuffer h_sieve_info failed. (clEnqueueReadBuffer)\n";
		return 1;
  }

  printArray("h_sieve_info", mystuff.h_sieve_info, info_size/sizeof(int), 1);

  info_size = mystuff.gpu_sieve_size / 8;

  status = clEnqueueReadBuffer(QUEUE,     // only for tracing/verification - not needed later.
                mystuff.d_bitarray,
                CL_TRUE,
                0,
                info_size,
                mystuff.h_bitarray,
                0,
                NULL,
                NULL);
    
  if(status != CL_SUCCESS) 
	{ 
    std::cout << "Error " << status << ": clEnqueueReadBuffer h_sieve_info failed. (clEnqueueReadBuffer)\n";
		return 1;
  }

  printArray("h_bitarray", mystuff.h_bitarray, info_size/sizeof(int), 1);
#endif

	return 0;
}

int run_mod_kernel(cl_ulong hi, cl_ulong lo, cl_ulong q, cl_float qr, cl_ulong *res_hi, cl_ulong *res_lo)
{
/* __kernel void mod_128_64_k(const ulong hi, const ulong lo, const ulong q, const float qr, __global ulong *res 
#if (TRACE_KERNEL > 1)
                  , __private uint tid
#endif
)
*/
  cl_int   status;
  cl_event mod_evt;

  *res_hi = *res_lo = 0;

  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    0, 
                    sizeof(cl_ulong), 
                    (void *)&hi);
  if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error " << status << ": Setting kernel argument. (hi)\n";
		return 1;
	}
  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    1, 
                    sizeof(cl_ulong), 
                    (void *)&lo);
  if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error " << status << ": Setting kernel argument. (lo)\n";
		return 1;
	}
  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    2, 
                    sizeof(cl_ulong), 
                    (void *)&q);
  if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error " << status << ": Setting kernel argument. (q)\n";
		return 1;
	}
  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    3, 
                    sizeof(cl_float), 
                    (void *)&qr);
  if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error " << status << ": Setting kernel argument. (qr)\n";
		return 1;
	}
  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    4, 
                    sizeof(cl_mem), 
                    (void *)&mystuff.d_RES);
  if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error " << status << ": Setting kernel argument. (RES)\n";
		return 1;
	}
  // dummy arg if KERNEL_TRACE is enabled: ignore errors if not.
  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    5, 
                    sizeof(cl_uint), 
                    (void *)&status);

  status = clEnqueueTask(QUEUE,
                 kernel_info[_TEST_MOD_].kernel,
                 0,
                 NULL,
                 &mod_evt);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<< "Error " << status << ": Enqueueing kernel(clEnqueueTask)\n";
		return 1;
	}

  status = clWaitForEvents(1, &mod_evt); 
  if(status != CL_SUCCESS) 
  { 
	  std::cerr<< "Error " << status << ": Waiting for mod call to finish. (clWaitForEvents)\n";
	  return 1;
  }
  #ifdef CL_PERFORMANCE_INFO
              cl_ulong startTime;
              cl_ulong endTime;
              /* Get kernel profiling info */
              status = clGetEventProfilingInfo(mod_evt,
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &startTime,
                                0);
              if(status != CL_SUCCESS)
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
                return 1;
              }
              status = clGetEventProfilingInfo(mod_evt,
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endTime,
                                0);
              if(status != CL_SUCCESS)
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
                return 1;
              }
              std::cout<< "mod_kernel finished in " << (endTime - startTime)/1e3 << " us.\n" ;
#endif

  status = clReleaseEvent(mod_evt);
  if(status != CL_SUCCESS) 
  { 
		std::cerr<< "Error " << status << ": Release mod event object. (clReleaseEvent)\n";
	  return 1;
  }
  status = clEnqueueReadBuffer(QUEUE,
                mystuff.d_RES,
                CL_TRUE,
                0,
                32 * sizeof(int),
                mystuff.h_RES,
                0,
                NULL,
                NULL);
    
  if(status != CL_SUCCESS) 
	{ 
    std::cout << "Error " << status << ": clEnqueueReadBuffer RES failed. (clEnqueueReadBuffer)\n";
		return 1;
  }
  *res_hi = mystuff.h_RES[0];
  *res_lo = mystuff.h_RES[1];

	return 0;

}

int run_kernel15(cl_kernel l_kernel, cl_uint exp, int75 k_base, int stream, cl_uint8 b_in, cl_mem res, cl_int shiftcount, cl_int bin_max)
/*
  run_kernel15(kernel_info[use_kernel].kernel, exp, k_base, i, b_in, mystuff->d_RES, shiftcount, bit_max);
*/
{
  cl_int   status;
  /*
__kernel void barrett15_73(__private uint exp, const int75_t k_base, const __global uint * restrict k_tab, const int shiftcount,
                           const uint8 b_in, __global uint * restrict RES, const int bit_max
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
*/
  //////// test test test ...
  // {k_min_grid[i] = 2822192209735ULL; mystuff.h_ktab[i][0]=0;}
  // k_base.d4=0;
  // k_base.d3=0;
  // k_base.d2=0xa44;
  // k_base.d1=0x2f86;
  // k_base.d0=0x7b2f;  // together with ktab[0]=2 this will run the proper factor in thread 0
  //new_class=1;
  ///////

  // first set the specific params that don't change per block: b_preinit, shiftcount, RES
  if (new_class)
  {
    status = clSetKernelArg(l_kernel, 
                    3, 
                    sizeof(cl_int), 
                    (void *)&shiftcount);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (shiftcount)\n";
  		return 1;
  	}

    status = clSetKernelArg(l_kernel, 
                    4, 
                    sizeof(cl_uint8),
                    (void *)&b_in);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (b_in)\n";
  		return 1;
  	}
      /* the bit_max for the barrett kernels (the others ignore it) */
      status = clSetKernelArg(l_kernel, 
                      6, 
                      sizeof(cl_int), 
                      (void *)&bin_max);
      if(status != CL_SUCCESS) 
      { 
        std::cerr<<"Warning " << status << ": Setting kernel argument. (bit_max)\n";
      }
#ifdef CHECKS_MODBASECASE
      status = clSetKernelArg(l_kernel, 
                    7, 
                    sizeof(cl_mem), 
                    (void *)&mystuff.d_modbasecase_debug);
      if(status != CL_SUCCESS) 
  	  { 
	  	  std::cerr<<"Error " << status << ": Setting kernel argument. (d_modbasecase_debug)\n";
  		  return 1;
  	  }
#endif
#ifdef DETAILED_INFO
    printf("run_kernel15: b=%x:%x:%x:%x:%x:%x:%x:%x:0:0, shift=%d\n",
      b_in.s[7], b_in.s[6], b_in.s[5], b_in.s[4], b_in.s[3], b_in.s[2], b_in.s[1], b_in.s[0], shiftcount);
#endif

  }
  // now the params that change everytime
  status = clSetKernelArg(l_kernel, 
                    1, 
                    sizeof(int75), 
                    (void *)&k_base);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<<"Error " << status << ": Setting kernel argument. (k_base)\n";
		return 1;
	}
#ifdef DETAILED_INFO
  printf("run_kernel15: k_base=%x:%x:%x:%x:%x\n", k_base.d4, k_base.d3, k_base.d2, k_base.d1, k_base.d0);
#endif
    
  return run_kernel(l_kernel, exp, stream, res); // set params 0,2,5 and start the kernel
}

int run_kernel24(cl_kernel l_kernel, cl_uint exp, int72 k_base, int stream, int144 b_preinit, cl_mem res, cl_int shiftcount, cl_int bin_min63)
/*
  run_kernel24(kernel_info[use_kernel].kernel, exp, k_base, i, b_preinit, mystuff->d_RES, shiftcount);
*/
{
  cl_int   status, argnum;
  /*
  __kernel void mfakto_cl_71(__private uint exp, __private int72_t k_base,
                             __global uint *k_tab, __private int shiftcount,
                             __private int144_t b, __global uint *RES)
   *

   __kernel void cl_barrett24_70(__private uint exp, const int72_t k_base, const __global uint * restrict k_tab, const int shiftcount,
#ifdef WA_FOR_CATALYST11_10_BUG
                           const uint8 b_in,
#else
                           __private int144_t bb,
#endif
                           __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )

*/
  //////// test test test ...
  // {k_min_grid[i] = 1777608657747ULL; mystuff->h_ktab[i][0]=0;}
  // k_base.d2=0;
  // k_base.d1=0;
  // k_base.d0=1;
  // new_class=1;
  ///////

  // first set the specific params that don't change per block: b_preinit, shiftcount, RES
  if (new_class)
  {
    status = clSetKernelArg(l_kernel, 
                    3, 
                    sizeof(cl_int), 
                    (void *)&shiftcount);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (shiftcount)\n";
  		return 1;
  	}
#ifdef WA_FOR_CATALYST11_10_BUG
    cl_uint8 b_in={{b_preinit.d0, b_preinit.d1, b_preinit.d2, b_preinit.d3, b_preinit.d4, b_preinit.d5, 0, 0}};
#endif

    status = clSetKernelArg(l_kernel, 
                    4, 
#ifdef WA_FOR_CATALYST11_10_BUG
                    sizeof(cl_uint8),
                    (void *)&b_in
#else
                    sizeof(int144), 
                    (void *)&b_preinit
#endif
        );
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (b_preinit)\n";
  		return 1;
  	}
    argnum=6;
    if ((kernel_info[BARRETT70_MUL24].kernel == l_kernel))
    {
      /* the bit_max-64 for the barrett kernels (the others ignore it) */
      status = clSetKernelArg(l_kernel, 
                      6, 
                      sizeof(cl_int), 
                      (void *)&bin_min63);
      if(status != CL_SUCCESS) 
      { 
        std::cerr<<"Warning " << status << ": Setting kernel argument. (bit_min)\n";
      }
      argnum=7;
  	}
#ifdef CHECKS_MODBASECASE
    if ((kernel_info[_71BIT_MUL24].kernel == l_kernel) || (kernel_info[_63BIT_MUL24].kernel == l_kernel) || (kernel_info[BARRETT70_MUL24].kernel == l_kernel))
    {
      status = clSetKernelArg(l_kernel, 
                    argnum, 
                    sizeof(cl_mem), 
                    (void *)&mystuff.d_modbasecase_debug);
      if(status != CL_SUCCESS) 
  	  { 
	  	  std::cerr<<"Error " << status << ": Setting kernel argument. (d_modbasecase_debug)\n";
  		  return 1;
  	  }
    }
#endif
#ifdef DETAILED_INFO
    printf("run_kernel24: b=%x:%x:%x:%x:%x:%x, shift=%d\n", b_preinit.d5, b_preinit.d4, b_preinit.d3, b_preinit.d2, b_preinit.d1, b_preinit.d0, shiftcount);
#endif

  }
  // now the params that change everytime
  status = clSetKernelArg(l_kernel, 
                    1, 
                    sizeof(int72), 
                    (void *)&k_base);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<<"Error " << status << ": Setting kernel argument. (k_base)\n";
		return 1;
	}
#ifdef DETAILED_INFO
  printf("run_kernel24: k_base=%x:%x:%x\n", k_base.d2, k_base.d1, k_base.d0);
#endif
    
  return run_kernel(l_kernel, exp, stream, res); // set params 0,2,5 and start the kernel
}

int run_kernel64(cl_kernel l_kernel, cl_uint exp, cl_ulong k_base, int stream, cl_ulong4 b_preinit, cl_mem res, cl_int bin_min63)
{
/*
 *	        cl_ulong4 b_preinit = {b_preinit_lo, b_preinit_mid, b_preinit_hi, shiftcount};
        run_kernel(kernel, exp, k_base, mystuff->d_ktab[stream], b_preinit, bit_min-63, mystuff->d_RES);
 */
  cl_int   status;
  /* __kernel void mfakto_cl_95(uint exp, ulong k, __global uint *k_tab, ulong4 b_pre_shift, int bit_max64, __global uint *RES) */
  // first set the specific params that don't change per block: b_pre_shift, bin_min63
  if (new_class)
  {
    status = clSetKernelArg(l_kernel, 
                    3, 
                    sizeof(cl_ulong4), 
                    (void *)&b_preinit);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (b_preinit)\n";
  		return 1;
  	}

    /* the bit_max-64 for the barrett kernels (the others ignore it) */
    status = clSetKernelArg(l_kernel, 
                    4, 
                    sizeof(cl_int), 
                    (void *)&bin_min63);
    if(status != CL_SUCCESS) 
  	{ 
	  	std::cerr<<"Warning " << status << ": Setting kernel argument. (bit_min)\n";
  		
  	}
#ifdef DETAILED_INFO
    printf("run_kernel64: b=%llx:%llx:%llx, shift=%u\n",
        (long long unsigned int)b_preinit.s[2], (long long unsigned int)b_preinit.s[1], (long long unsigned int)b_preinit.s[0], (unsigned int)b_preinit.s[3]);
#endif
  }
  // now the params that change everytime
  status = clSetKernelArg(l_kernel, 
                    1, 
                    sizeof(cl_ulong), 
                    (void *)&k_base);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<<"Error " << status << ": Setting kernel argument. (k_base)\n";
		return 1;
	}
#ifdef DETAILED_INFO
  printf("run_kernel64: kbase=%llu\n", (long long unsigned int) k_base);
#endif
  return run_kernel(l_kernel, exp, stream, res);
}

int run_barrett_kernel32(cl_kernel l_kernel, cl_uint exp, int96 k_base, int stream,
                  int192 b_preinit, cl_mem res, cl_int shiftcount, cl_int bin_min63)
{
  cl_int   status;
  /* __kernel void cl_barrett32_79(__private uint exp, __private int96 k, __global uint *k_tab,
         __private int shiftcount, __private int192_v b, __global uint *RES, __private int bit_max64) */
  // first set the specific params that don't change per block: b_pre_shift, bin_min63
  if (new_class)
  {
    status = clSetKernelArg(l_kernel, 
                    3, 
                    sizeof(cl_int), 
                    (void *)&shiftcount);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (shiftcount)\n";
  		return 1;
  	}

#ifdef WA_FOR_CATALYST11_10_BUG
    cl_uint8 b_in={{b_preinit.d0, b_preinit.d1, b_preinit.d2, b_preinit.d3, b_preinit.d4, b_preinit.d5, 0, 0}};
#endif

    status = clSetKernelArg(l_kernel, 
                    4, 
#ifdef WA_FOR_CATALYST11_10_BUG
                    sizeof(cl_uint8),
                    (void *)&b_in
#else
                    sizeof(int144), 
                    (void *)&b_preinit
#endif
        );
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (b_preinit)\n";
  		return 1;
  	}

    /* the bit_max-64 for the barrett kernels (the others ignore it) */
    status = clSetKernelArg(l_kernel, 
                    6, 
                    sizeof(cl_int), 
                    (void *)&bin_min63);
    if(status != CL_SUCCESS) 
  	{ 
	  	std::cerr<<"Warning " << status << ": Setting kernel argument. (bit_min)\n";
  		
  	}
#ifdef CHECKS_MODBASECASE
    status = clSetKernelArg(l_kernel, 
                    7, 
                    sizeof(cl_mem), 
                    (void *)&mystuff.d_modbasecase_debug);
    if(status != CL_SUCCESS) 
	  { 
   	  std::cerr<<"Error " << status << ": Setting kernel argument. (d_modbasecase_debug)\n";
		  return 1;
	  }
#endif

#ifdef DETAILED_INFO
    printf("run_barrett_kernel32: b=%x:%x:%x:%x:%x:%x, shift=%d\n", b_preinit.d5, b_preinit.d4,
        b_preinit.d3, b_preinit.d2, b_preinit.d1, b_preinit.d0, shiftcount);
#endif
  }
  // now the params that change everytime
  status = clSetKernelArg(l_kernel, 
                    1, 
                    sizeof(int96), 
                    (void *)&k_base);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<<"Error " << status << ": Setting kernel argument. (k_base)\n";
		return 1;
	}
#ifdef DETAILED_INFO
  printf("run_barrett_kernel32: kbase=%x:%x:%x\n", k_base.d2, k_base.d1, k_base.d0);
#endif
  return run_kernel(l_kernel, exp, stream, res);
}

int run_kernel(cl_kernel l_kernel, cl_uint exp, int stream, cl_mem res)
{
  cl_int   status;
  cl_mem   k_tab = mystuff.d_ktab[stream];
  size_t   globalThreads[2];
  size_t   localThreads[2];
  size_t   total_threads = mystuff.threads_per_grid;

  // adjust for vector kernels: each thread processes 1-16 FC's, use accordingly less threads
  total_threads /= mystuff.vectorsize;

  globalThreads[0] = (total_threads > deviceinfo.maxThreadsPerBlock) ? deviceinfo.maxThreadsPerBlock : total_threads;
  globalThreads[1] = (total_threads-1)/deviceinfo.maxThreadsPerBlock + 1;
  localThreads[0] = globalThreads[0];  // PERF: test different sizes, also in combination with the __attribute__((reqd_work_group_size(X, Y, Z)))qualifier
  localThreads[1] = 1;

  // first set the params that don't change per block: exp, RES
  if (new_class)
  {
    status = clSetKernelArg(l_kernel, 
                    0, 
                    sizeof(cl_uint), 
                    (void *) &exp);
    if(status != CL_SUCCESS) 
	  { 
	  	std::cerr<<"Error " << status << ": Setting kernel argument. (exp)\n";
	  	return 1;
  	}

    /* the output array to the kernel */
    status = clSetKernelArg(l_kernel, 
                    5, 
                    sizeof(cl_mem), 
                    (void *)&res);
    if(status != CL_SUCCESS) 
  	{ 
	  	std::cerr<<"Error " << status << ": Setting kernel argument. (res)\n";
  		return 1;
  	}

    new_class = 0; // do not set these params again until a new class is started
  }

  status = clSetKernelArg(l_kernel, 
                    2, 
                    sizeof(cl_mem), 
                    (void *)&k_tab);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<<"Error " << status << ": Setting kernel argument. (k_tab)\n";
		return 1;
	}

  status = clEnqueueNDRangeKernel(QUEUE,
                 l_kernel,
                 2,
                 NULL,
                 globalThreads,
                 localThreads,
                 1,
                 &mystuff.copy_events[stream], // wait for the k_tab write to finish
                 &mystuff.exec_events[stream]);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<< "Error " << status << ": Enqueuing kernel(clEnqueueNDRangeKernel), stream " << stream << "\n";
		return 1;
	}
  clFlush(QUEUE);
	return 0;
}

int run_gs_kernel15(cl_kernel kernel, cl_uint numblocks, cl_uint shared_mem_required, int75 k_base, cl_uint8 b_in, cl_uint shiftcount)
{
  cl_int   status;
  /*
__kernel void cl_barrett32_77_gs(__private uint exp, const int96_t k_base, const __global uint * restrict bit_array, const uint bits_to_process, __local ushort *smem, const int shiftcount,
                           __private int192_t bb, __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
*/
  // first set the specific params that don't change per block: b_in
  if (new_class)
  {
    status = clSetKernelArg(kernel, 
                    6, 
                    sizeof(cl_uint8), 
                    (void *)&b_in
        );
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (b_in)\n";
  		return 1;
  	}
  }
#ifdef DETAILED_INFO
    printf("run_gs_kernel15: b=%x:%x:%x:%x:%x:%x:%x:%x, shift=%d\n",
      b_in.s[0], b_in.s[1], b_in.s[2], b_in.s[3], b_in.s[4], b_in.s[5], b_in.s[6], b_in.s[7], shiftcount);
#endif

  // now the params that change everytime
  status = clSetKernelArg(kernel, 
                    1, 
                    sizeof(int75), 
                    (void *)&k_base);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<<"Error " << status << ": Setting kernel argument. (k_base)\n";
		return 1;
	}
#ifdef DETAILED_INFO
  printf("run_gs_kernel15: k_base=%x:%x:%x\n", k_base.d2, k_base.d1, k_base.d0);
#endif
    
  return run_gs_kernel(kernel, numblocks, shared_mem_required, shiftcount);
}

int run_gs_kernel32(cl_kernel kernel, cl_uint numblocks, cl_uint shared_mem_required, int96 k_base, int192 b_preinit, cl_uint shiftcount)
{
  cl_int   status;
  /*
__kernel void cl_barrett32_77_gs(__private uint exp, const int96_t k_base, const __global uint * restrict bit_array, const uint bits_to_process, __local ushort *smem, const int shiftcount,
                           __private int192_t bb, __global uint * restrict RES, const int bit_max64
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
*/
  // first set the specific params that don't change per block: b_in
#ifdef WA_FOR_CATALYST11_10_BUG
    cl_uint8 b_in={{b_preinit.d0, b_preinit.d1, b_preinit.d2, b_preinit.d3, b_preinit.d4, b_preinit.d5, 0, 0}};
#endif

  if (new_class)
  {
    status = clSetKernelArg(kernel, 
                    6, 
#ifdef WA_FOR_CATALYST11_10_BUG
                    sizeof(cl_uint8),
                    (void *)&b_in
#else
                    sizeof(int192), 
                    (void *)&b_preinit
#endif
        );
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (b_in)\n";
  		return 1;
  	}
  }
#ifdef DETAILED_INFO
    printf("run_gs_kernel32: b=%x:%x:%x:%x:%x:%x, shift=%d\n",
      b_preinit.d5, b_preinit.d4, b_preinit.d3, b_preinit.d2, b_preinit.d1, b_preinit.d0, shiftcount);
#endif

  // now the params that change everytime
  status = clSetKernelArg(kernel, 
                    1, 
                    sizeof(int96), 
                    (void *)&k_base);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr<<"Error " << status << ": Setting kernel argument. (k_base)\n";
		return 1;
	}
#ifdef DETAILED_INFO
  printf("run_gs_kernel32: k_base=%x:%x:%x\n", k_base.d2, k_base.d1, k_base.d0);
#endif
    
  return run_gs_kernel(kernel, numblocks, shared_mem_required, shiftcount);
}

/* set all generic parameters for GPU-sieve-aware TF kernels and start them */
int run_gs_kernel(cl_kernel kernel, cl_uint numblocks, cl_uint shared_mem_required, cl_uint shiftcount)
{
  /*
__kernel void cl_barrett32_77_gs(__private uint exp, const int96_t k_base, const __global uint * restrict bit_array, const uint bits_to_process, __local ushort *smem, const int shiftcount,
                           __private int192_t bb, __global uint * restrict RES, const int bit_max64, const uint shared_mem_allocated
#ifdef CHECKS_MODBASECASE
         , __global uint * restrict modbasecase_debug
#endif
         )
*/
  // params 1 (k_base) and 6 (bb) are already set when entering this function
  cl_int   status;
  cl_event run_event;
  size_t   globalThreads=numblocks*256;
  size_t   localThreads=256;

//  shared_mem_required = (shared_mem_required + 127) & 0xFFFFFF80; // 128-byte-multiple
#ifdef DETAILED_INFO
  printf("run_gs_kernel: shared_mem: %u, loc/glob threads: %u/%u\n", shared_mem_required, localThreads, globalThreads);
#endif
  if (new_class)
  {
    new_class = 0;
    status = clSetKernelArg(kernel, 
                    0, 
                    sizeof(cl_uint), 
                    (void *)&mystuff.exponent);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (exponent)\n";
  		return 1;
  	}

    status = clSetKernelArg(kernel, 
                    2, 
                    sizeof(cl_mem), 
                    (void *)&mystuff.d_bitarray);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (d_bitarray)\n";
  		return 1;
  	}

    status = clSetKernelArg(kernel, 
                    3, 
                    sizeof(cl_uint), 
                    (void *)&mystuff.gpu_sieve_processing_size);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (gpu_sieve_processing_size)\n";
  		return 1;
  	}

    status = clSetKernelArg(kernel, 
                    4, 
                    shared_mem_required, 
                    NULL);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (smem)\n";
  		return 1;
  	}

    status = clSetKernelArg(kernel, 
                    5, 
                    sizeof(cl_uint), 
                    (void *)&shiftcount);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (gpu_sieve_processing_size)\n";
  		return 1;
  	}

    status = clSetKernelArg(kernel, 
                    7, 
                    sizeof(cl_mem), 
                    (void *)&mystuff.d_RES);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (d_bitarray)\n";
  		return 1;
  	}

    cl_uint tmp = mystuff.bit_max_stage - 64;

    status = clSetKernelArg(kernel, 
                    8, 
                    sizeof(cl_uint), 
                    (void *)&tmp);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (bit_max64)\n";
  		return 1;
  	}

    status = clSetKernelArg(kernel, 
                    9, 
                    sizeof(cl_uint), 
                    (void *)&shared_mem_required);
    if(status != CL_SUCCESS) 
  	{ 
  		std::cerr<< "Error " << status << ": Setting kernel argument. (shared_mem_required)\n";
  		return 1;
  	}

#ifdef CHECKS_MODBASECASE
      status = clSetKernelArg(l_kernel, 
                    10, 
                    sizeof(cl_mem), 
                    (void *)&mystuff.d_modbasecase_debug);
      if(status != CL_SUCCESS) 
  	  { 
	  	  std::cerr<<"Error " << status << ": Setting kernel argument. (d_modbasecase_debug)\n";
  		  return 1;
  	  }
#endif
  }
  // all set? now start the kernel
  status = clEnqueueNDRangeKernel(QUEUE,
                 kernel,
                 1,
                 NULL,
                 &globalThreads,
                 &localThreads,
                 0,
                 NULL,
#ifdef CL_PERFORMANCE_INFO
                 &run_event
#else
                 NULL
#endif
                 ); // no need to wait for anything - they will be processed serially, and we read the results synchronously.

  if(status != CL_SUCCESS) 
	{ 
		std::cerr<< "Error " << status << ": Enqueuing kernel(clEnqueueNDRangeKernel)\n";
		return 1;
	}
  clFlush(QUEUE);
#ifdef CL_PERFORMANCE_INFO
  clFinish(QUEUE);
  cl_ulong startTime=0;  // device time in nanosecs
  cl_ulong endTime=1000;
  /* Get kernel profiling info */
  status = clGetEventProfilingInfo(run_event,
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &startTime,
                                0);
  if(status != CL_SUCCESS)
 	{ 
		std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
    return RET_ERROR;
  }
  status = clGetEventProfilingInfo(run_event,
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endTime,
                                0);
  if(status != CL_SUCCESS) 
 	{ 
		std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
    return RET_ERROR;
  }
  std::cout<< "TF using " << globalThreads << " threads: " << (endTime - startTime)/1e6 << " ms ("
                       << globalThreads * 1e3 / (endTime - startTime) << " M/s), " <<
                       mystuff.gpu_sieve_size * 1e3 / (endTime - startTime) << " M FCs/s TF'd (after sieving)\n";
  clReleaseEvent(mystuff.copy_events[0]);
#endif

	return 0;
}


int cleanup_CL(void)
{
  cl_int status;
  cl_uint i;

  for (i=0; i<NUM_KERNELS; i++)
  {
    if (kernel_info[i].kernel)
    {
      status = clReleaseKernel(kernel_info[i].kernel);
      if(status != CL_SUCCESS)
    	{
	    	fprintf(stderr, "Error %d: clReleaseKernel(%d)\n", status, i);
	    	return 1; 
      }
  	}
  }

  status = clReleaseProgram(program);
  if(status != CL_SUCCESS)
	{
		std::cerr<<"Error" << status << ": clReleaseProgram\n";
		return 1; 
	}
  for (i=0; i<mystuff.num_streams; i++)
  {
    status = clReleaseMemObject(mystuff.d_ktab[i]);
    if(status != CL_SUCCESS)
  	{
	  	std::cerr<<"Error" << status << ": clReleaseMemObject (d_ktab" << i << ")\n";
  		return 1; 
  	}
    free(mystuff.h_ktab[i]);
  }
	status = clReleaseMemObject(mystuff.d_RES);
  if(status != CL_SUCCESS)
	{
		std::cerr<<"Error" << status << ": clReleaseMemObject (d_RES)\n";
		return 1; 
	}
  free(mystuff.h_RES);
#ifdef CHECKS_MODBASECASE
	status = clReleaseMemObject(mystuff.d_modbasecase_debug);
  if(status != CL_SUCCESS)
	{
		std::cerr<<"Error" << status << ": clReleaseMemObject (d_modbasecase_debug)\n";
		return 1; 
	}
  free(mystuff.h_modbasecase_debug);
#endif
  if (mystuff.gpu_sieving == 1)
  {
    gpusieve_free (&mystuff);
  }

  status = clReleaseCommandQueue(commandQueue);
  if(status != CL_SUCCESS)
	{
		std::cerr<<"Error" << status << ": clReleaseCommandQueue\n";
		return 1;
	}
  if (commandQueuePrf) status = clReleaseCommandQueue(commandQueuePrf);
  if(status != CL_SUCCESS)
	{
		std::cerr<<"Error" << status << ": clReleaseCommandQueuePrf\n";
		return 1;
	  status = clReleaseContext(context);
  }
  if(status != CL_SUCCESS)
	{
		std::cerr<<"Error" << status << ": clReleaseContext\n";
		return 1;
	}
  if(devices != NULL)
  {
      free(devices);
      devices = NULL;
  }
	return 0;
}

int tf_class_opencl(cl_ulong k_min, cl_ulong k_max, mystuff_t *mystuff, enum GPUKernels use_kernel)
{
  size_t size = mystuff->threads_per_grid * sizeof(int);
  int status, wait = 0;
  struct timeval timer, timer2;
  cl_ulong twait=0;
  cl_uint cwait=0, i;
// for TF_72BIT  
  int72  k_base;
  int144 b_preinit = {0};
  int192 b_192 = {0};
  cl_uint8 b_in = {{0}};
  int96  factor;

  cl_uint  factorsfound=0;
  cl_uint  shiftcount, ln2b, count=1, shared_mem_required, numblocks;
  cl_ulong b_preinit_lo, b_preinit_mid, b_preinit_hi;
  cl_ulong k_diff, k_remaining;
  char string[50];
  int running=0;
  
  int h_ktab_index = 0;
  unsigned long long int k_min_grid[NUM_STREAMS_MAX];	// k_min_grid[N] contains the k_min for h_ktab[N], only valid for preprocessed h_ktab[]s

  timer_init(&timer);
#ifdef DETAILED_INFO
  printf("tf_class_opencl(%u, %d, %llu, %llu, ...)\n",
      mystuff->exponent, mystuff->bit_min, (long long unsigned int) k_min, (long long unsigned int) k_max);
#endif

  //  mystuff->exponent=51152869; k_min=20582854459640ULL; k_max=20582854459641ULL;  // test test test

  new_class=1; // tell run_kernel to re-submit the one-time kernel arguments
  if ( k_max <= k_min) k_max = k_min + 1;  // otherwise it would skip small bit ranges

  /* set result array to 0 */ 
  memset(mystuff->h_RES,0,32 * sizeof(int));
  status = clEnqueueWriteBuffer(QUEUE,
                mystuff->d_RES,
                CL_TRUE,          // Wait for completion; it's fast to copy 128 bytes ;-)
                0,
                32 * sizeof(int),
                mystuff->h_RES,
                0,
                NULL,
                &mystuff->copy_events[0]);
  if(status != CL_SUCCESS) 
	{  
		std::cout<<"Error " << status << ": Copying h_RES(clEnqueueWriteBuffer)\n";
		return RET_ERROR; // # factors found ;-)
	}
#ifdef CHECKS_MODBASECASE
  /* set modbasecase_debug array to 0 */ 
  memset(mystuff->h_modbasecase_debug,0,32 * sizeof(int));
  status = clEnqueueWriteBuffer(QUEUE,
                mystuff->d_modbasecase_debug,
                CL_TRUE,
                0,
                32 * sizeof(int),
                mystuff->h_modbasecase_debug,
                0,
                NULL,
                NULL);
  if(status != CL_SUCCESS) 
	{  
		std::cout<<"Error " << status << ": Copying h_modbasecase_debug(clEnqueueWriteBuffer)\n";
		return RET_ERROR; // # factors found ;-)
	}
#endif

  for(i=0; i<mystuff->num_streams; i++)
  {
    mystuff->stream_status[i] = UNUSED;
    k_min_grid[i] = 0;
  }
  
  shiftcount=10;  // no exp below 2^10 ;-)
  while((1ULL<<shiftcount) < (unsigned long long int)mystuff->exponent)shiftcount++;
#ifdef DETAILED_INFO
  printf("bits in exp %u: %u, ", mystuff->exponent, shiftcount);
#endif
  shiftcount -= 6; // all kernels can handle 5 bits of pre-shift (max 2^63)
  ln2b = mystuff->exponent >> shiftcount;
  // some kernels may actually accept a higher preprocessed value
  // but it's hard to find the exact limit, and mfakto already had
  // a bug with the precalculation being too high
  // Therefore: play it safe and just precalc as far as the algorithm including modulus would go

  while (ln2b < mystuff->bit_max_stage)
  {
    shiftcount--;
    ln2b = mystuff->exponent >> shiftcount;
  }
#ifdef DETAILED_INFO
  printf("remaining shiftcount = %d, ln2b = %d\n", shiftcount, ln2b);
#endif
  b_preinit_hi=0;b_preinit_mid=0;b_preinit_lo=0;
  count=0;
// set the pre-initriables in all sizes for all possible kernels
  {
    if     (ln2b<24 ){fprintf(stderr, "Pre-init (%u) too small\n", ln2b); return RET_ERROR;}      // should not happen
    else if(ln2b<48 )b_preinit.d1=1<<(ln2b-24);   // should not happen
    else if(ln2b<72 )b_preinit.d2=1<<(ln2b-48);
    else if(ln2b<96 )b_preinit.d3=1<<(ln2b-72);
    else if(ln2b<120)b_preinit.d4=1<<(ln2b-96);
    else             b_preinit.d5=1<<(ln2b-120);	// b_preinit = 2^ln2b
  }

  { // skip the "lowest" 4 levels, so that uint8 is sufficient for 12 components of int180
    if     (ln2b<60 ){fprintf(stderr, "Pre-init (%u) too small\n", ln2b); return RET_ERROR;}      // should not happen
    else if(ln2b<75 )b_in.s[0]=1<<(ln2b-60);
    else if(ln2b<90 )b_in.s[1]=1<<(ln2b-75);
    else if(ln2b<105)b_in.s[2]=1<<(ln2b-90);
    else if(ln2b<120)b_in.s[3]=1<<(ln2b-105);
    else if(ln2b<135)b_in.s[4]=1<<(ln2b-120);
    else if(ln2b<150)b_in.s[5]=1<<(ln2b-135);
    else if(ln2b<165)b_in.s[6]=1<<(ln2b-150);
    else             b_in.s[7]=1<<(ln2b-165);
  }

  {
    if     (ln2b<32 )b_192.d0=1<< ln2b;       // should not happen
    else if(ln2b<64 )b_192.d1=1<<(ln2b-32);   // should not happen
    else if(ln2b<96 )b_192.d2=1<<(ln2b-64);
    else if(ln2b<128)b_192.d3=1<<(ln2b-96);
    else if(ln2b<160)b_192.d4=1<<(ln2b-128);
    else             b_192.d5=1<<(ln2b-160);	// b_preinit = 2^ln2b
  }

  {
    if     (ln2b<64 )b_preinit_lo = 1ULL<< ln2b;
    else if(ln2b<128)b_preinit_mid= 1ULL<<(ln2b-64);
    else             b_preinit_hi = 1ULL<<(ln2b-128); // b_preinit = 2^ln2b
  }

  // combine for more efficient passing of parameters
  cl_ulong4 b_preinit4 = {{b_preinit_lo, b_preinit_mid, b_preinit_hi, (cl_ulong)shiftcount-1}};
#ifdef RAW_GPU_BENCH
  shared_mem_required = 100;						// no sieving = 100%
#else
  if (mystuff->gpu_sieve_primes < 54) shared_mem_required = 100;	// no sieving = 100%
  else if (mystuff->gpu_sieve_primes < 310) shared_mem_required = 50;	// 54 primes expect 48.30%
  else if (mystuff->gpu_sieve_primes < 1846) shared_mem_required = 38;	// 310 primes expect 35.50%
  else if (mystuff->gpu_sieve_primes < 21814) shared_mem_required = 30;	// 1846 primes expect 28.10%
  else if (mystuff->gpu_sieve_primes < 67894) shared_mem_required = 24;	// 21814 primes expect 21.93%
  else shared_mem_required = 22;					// 67894 primes expect 19.94%
#endif
  shared_mem_required = mystuff->gpu_sieve_processing_size * sizeof (short) * shared_mem_required / 100;

  while((k_min <= k_max) || (running > 0))
  {
    h_ktab_index = count % mystuff->num_streams;

/* preprocessing: calculate a ktab (factor table) */
    if((mystuff->stream_status[h_ktab_index] == UNUSED) && (k_min <= k_max))	// if we have an empty h_ktab we can preprocess another one
    {
#ifdef DEBUG_STREAM_SCHEDULE
      printf(" STREAM_SCHEDULE: preprocessing on h_ktab[%d]\n", h_ktab_index);
#endif

      if (mystuff->gpu_sieving == 0)
      {
        sieve_candidates(mystuff->threads_per_grid, mystuff->h_ktab[h_ktab_index], mystuff->sieve_primes);
        k_diff=mystuff->h_ktab[h_ktab_index][mystuff->threads_per_grid-1]+1;
        k_diff*=NUM_CLASSES;				/* NUM_CLASSES because classes are mod NUM_CLASSES */
      
        k_min_grid[h_ktab_index] = k_min;
        /* try upload ktab*/

        status = clEnqueueWriteBuffer(QUEUE,
                  mystuff->d_ktab[h_ktab_index],
                  CL_FALSE,
                  0,
                  size,
                  mystuff->h_ktab[h_ktab_index],
                  0,
                  NULL,
                  &mystuff->copy_events[h_ktab_index]);

        if(status != CL_SUCCESS) 
	      {  
	          std::cout<<"Error " << status << ": Copying h_ktab(clEnqueueWriteBuffer)\n";
            return RET_ERROR; // # factors found ;-)
	      }
      }
      else
      {
        // GPU sieving
        // Calculate the number of k's remaining.  Round this up so that we sieve an array that is
        // a multiple of the bits processed by each TF kernel (my_stuff->gpu_sieve_processing_size).

        k_remaining = ((k_max - k_min + 1) + NUM_CLASSES - 1) / NUM_CLASSES;
        if (k_remaining < (cl_ulong) mystuff->gpu_sieve_size) {
          numblocks = (cl_uint) ((k_remaining + mystuff->gpu_sieve_processing_size - 1) / mystuff->gpu_sieve_processing_size);
          k_remaining = numblocks * mystuff->gpu_sieve_processing_size;
        } else
          numblocks = mystuff->gpu_sieve_size / mystuff->gpu_sieve_processing_size;

        // the sieving

        gpusieve (mystuff, k_max-k_min);

#ifdef DETAILED_INFO
  // as a first test, copy the sieve bits into the usual sieve array - later, the kernels will do that.
        cl_uint ind=0, pos=0, *dest=mystuff->h_ktab[h_ktab_index];
        for (i=0; i<mystuff->gpu_sieve_size/32 && ind < mystuff->threads_per_grid; i++)
        {
          cl_uint ii, word=mystuff->h_bitarray[i];
          for (ii=0; ii<32 && ind < mystuff->threads_per_grid; ii++)
          {
            if (word & 1)
            {
              dest[ind++]=pos;
              // simple verify ...
/*              for (cl_uint p=13; p<mystuff->gpu_sieve_primes; p+=2)
              {
                cl_ulong rem=k_min%p + ((cl_ulong)pos*NUM_CLASSES)%p;
                rem=(2*mystuff->exponent*rem +1)%p;
                if (rem == 0)
                {
                  //if (p>512)
                  {
                    printf("pos %d: %d ==> f divisible by %d\n", ind, pos, p);
                  }
                  break;
                }
              }*/
            }
            pos++;
            if (pos > 0xffffff) printf("Overflow!\n");
            word >>= 1;
          }
        }
        printf("bit-extract: %u/%u words processed, %u(max %u) bits set.\n", i, mystuff->gpu_sieve_size/32, ind, mystuff->threads_per_grid);
#endif
        // Now let the GPU trial factor the candidates that survived the sieving

        if (use_kernel >= BARRETT73_MUL15_GS && use_kernel <= BARRETT82_MUL15_GS)
        {
          int75 k_base;
          k_base.d0 =  k_min & 0x7FFF;
          k_base.d1 = (k_min >> 15) & 0x7FFF;
          k_base.d2 = (k_min >> 30) & 0x7FFF;
          k_base.d3 = (k_min >> 45) & 0x7FFF;
          k_base.d4 =  k_min >> 60;
          status = run_gs_kernel15(kernel_info[use_kernel].kernel, numblocks, shared_mem_required, k_base, b_in, shiftcount);
        }
        else if (use_kernel >= BARRETT79_MUL32_GS && use_kernel <= BARRETT87_MUL32_GS)
        {
          int96 k_base;
          k_base.d0 = (cl_uint) k_min;
          k_base.d1 = k_min >> 32;
          k_base.d2 = 0;
          status = run_gs_kernel32(kernel_info[use_kernel].kernel, numblocks, shared_mem_required, k_base, b_192, shiftcount);
        }

        // Count the number of blocks processed
        count += numblocks;

        // Move to next batch of k's
        k_min += (cl_ulong) mystuff->gpu_sieve_size * NUM_CLASSES;
        if (k_min > k_max) break;

        //BUG - we should call a different routine to advance the bit-to-clear values by gpusieve_size bits
        // This will be cheaper than recomputing the bit-to-clears from scratch
        // HOWEVER, the self-test code will not check this new code unless we make the gpusieve_size much smaller
        gpusieve_init_class (mystuff, k_min);
        continue; // don't go to the stream-scheduling code below - the GPU sieve runs the TF kernels all in one stream
      }
	    mystuff->stream_status[h_ktab_index] = PREPARED;
      running++;
#ifdef DETAILED_INFO
      printf("k-base: %llu, ", (long long unsigned int) k_min);
      printArray("ktab", mystuff->h_ktab[h_ktab_index], mystuff->threads_per_grid, 0);
#endif

      count++;
      k_min += (unsigned long long int)k_diff;
    }

    wait = 1;

    for(i=0; i<mystuff->num_streams; i++)
    {
      switch (mystuff->stream_status[i])
      {
        case UNUSED:
          {
            if (k_min <= k_max)
            {
              wait = 0;
            }
            break;  // still some work to do
            // continue; // check the other streams
          }
        case PREPARED:                   // start the calculation of a preprocessed dataset on the device
          {
            if ((use_kernel == _71BIT_MUL24) || (use_kernel == _63BIT_MUL24) || (use_kernel == BARRETT70_MUL24))
            {
              k_base.d0 =  k_min_grid[i] & 0xFFFFFF;
              k_base.d1 = (k_min_grid[i] >> 24) & 0xFFFFFF;
              k_base.d2 =  k_min_grid[i] >> 48;
              status = run_kernel24(kernel_info[use_kernel].kernel, mystuff->exponent, k_base, i, b_preinit, mystuff->d_RES, shiftcount, mystuff->bit_min-63);
            }
            else if (((use_kernel >= BARRETT73_MUL15) && (use_kernel <= BARRETT82_MUL15)) || (use_kernel == MG88))
            {
              int75 k_base;
              k_base.d0 =  k_min_grid[i] & 0x7FFF;
              k_base.d1 = (k_min_grid[i] >> 15) & 0x7FFF;
              k_base.d2 = (k_min_grid[i] >> 30) & 0x7FFF;
              k_base.d3 = (k_min_grid[i] >> 45) & 0x7FFF;
              k_base.d4 =  k_min_grid[i] >> 60;
              status = run_kernel15(kernel_info[use_kernel].kernel, mystuff->exponent, k_base, i, b_in, mystuff->d_RES, shiftcount, mystuff->bit_max_stage);
            }
            else if (((use_kernel >= BARRETT79_MUL32) && (use_kernel <= BARRETT87_MUL32)) || (use_kernel == MG62))
            {
              int96 k;
              k.d0 = (cl_uint) k_min_grid[i];
              k.d1 = k_min_grid[i] >> 32;
              k.d2 = 0;
              status = run_barrett_kernel32(kernel_info[use_kernel].kernel, mystuff->exponent, k, i, b_192, mystuff->d_RES, shiftcount, mystuff->bit_min-63);
            }
            else
            {
              status = run_kernel64(kernel_info[use_kernel].kernel, mystuff->exponent, k_min_grid[i], i, b_preinit4, mystuff->d_RES, mystuff->bit_min-63);
            }
            if(status != CL_SUCCESS) 
            { 
	            std::cerr<< "Error " << status << ": Starting kernel " << kernel_info[use_kernel].kernelname << ". (run_kernel)\n";
	    	      return RET_ERROR;
            }

#ifdef DEBUG_STREAM_SCHEDULE
            printf(" STREAM_SCHEDULE: started GPU kernel using h_ktab[%d] (%s, %u, %llu, ...)\n", i, kernel_info[use_kernel].kernelname, mystuff->exponent, k_min_grid[i]);
#endif
            mystuff->stream_status[i] = RUNNING;
            break;
            // continue; // examine the next stream
          }
        case RUNNING:                    // check if it really is still running
          {
            cl_int event_status;
            status = clGetEventInfo(mystuff->exec_events[i],
                         CL_EVENT_COMMAND_EXECUTION_STATUS,
                         sizeof(cl_int),
                         &event_status,
                         NULL);
#ifdef DEBUG_STREAM_SCHEDULE
	          std::cout<<  " STREAM_SCHEDULE: Querying event " << i << " = " << event_status << "\n";
#endif
            if(status != CL_SUCCESS) 
            { 
	            std::cerr<< "Error " << status << ": Querying event " << i << ". (clGetEventInfo)\n";
	    	      return RET_ERROR;
            }
            if (event_status > CL_COMPLETE) /* still running: CL_QUEUED=3 (command has been enqueued in the command-queue),
                                               CL_SUBMITTED=2 (enqueued command has been submitted by the host to the
                                               device associated with the command-queue),
                                               CL_RUNNING=1 (device is currently executing this command), CL_COMPLETE=0,
                                               any error: <0 */
            {
              break;
              // continue; // examine the next stream
            }
            else // finished
            {
#ifdef CL_PERFORMANCE_INFO
              cl_ulong startTime=0;
              cl_ulong endTime=1000;
              /* Get kernel profiling info */
              if (use_kernel != _95BIT_64_OpenCL && !mystuff->gpu_sieving) 
              {
                status = clGetEventProfilingInfo(mystuff->copy_events[i],
                                  CL_PROFILING_COMMAND_START,
                                  sizeof(cl_ulong),
                                  &startTime,
                                  0);
                if(status != CL_SUCCESS)
                { 
                  std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
                  return RET_ERROR;
                }
                status = clGetEventProfilingInfo(mystuff->copy_events[i],
                                  CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong),
                                  &endTime,
                                  0);
                if(status != CL_SUCCESS) 
                { 
                  std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
                  return RET_ERROR;
                }
                printf("%d FCs copied in %2.2f ms (%4.2f MB/s), ", mystuff->threads_per_grid, (endTime - startTime)/1e6,
                        size * 1e3 / (endTime - startTime) );
              }
              status = clGetEventProfilingInfo(mystuff->exec_events[i],
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &startTime,
                                0);
              if(status != CL_SUCCESS)
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
                return RET_ERROR;
              }
              status = clGetEventProfilingInfo(mystuff->exec_events[i],
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endTime,
                                0);
              if(status != CL_SUCCESS)
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
                return RET_ERROR;
              }
              printf("proc'd in %2.2f ms (%3.2f M/s)\n", (endTime - startTime)/1e6, double(mystuff->threads_per_grid) *1e3/ (endTime - startTime));
#endif
              status = clReleaseEvent(mystuff->exec_events[i]);
              if(status != CL_SUCCESS) 
              { 
		         	  std::cerr<< "Error " << status << ": Release exec event object. (clReleaseEvent)\n";
		         	  return RET_ERROR;
    	       	}
              if (!mystuff->gpu_sieving) status = clReleaseEvent(mystuff->copy_events[i]);
             	if(status != CL_SUCCESS) 
           	  { 
		          	std::cerr<< "Error " << status << ": Release copy event object. (clReleaseEvent)\n";
		         	  return RET_ERROR;
            	}

              if (event_status < CL_COMPLETE) // error
              {
	              std::cerr<< "Error " << event_status << " during execution of block " << count << " in h_ktab[" << i << "]\n";
	    	        return RET_ERROR;
              }
              else
              {
                mystuff->stream_status[i] = DONE;
                /* no break to fall through to process the DONE value */
              }
            }
          }
        case DONE:                       // get the results
          {                              // or maybe not; wait until the class is done.
            mystuff->stream_status[i] = UNUSED;
            --running;
            if ((k_min <= k_max) || (running==0))
            {
              wait = 0;  // some k's left to be processed, or nothing running on GPU - not time to sleep!
            }
            break;
            // continue; // check the other streams
          }
      }
     // break; // out of the loop as we can process another stream (shortcut: don't check the other streams now)
    }

    if(wait > 0)
    {
      /* no unused h_ktab for preprocessing. 
      This usually means that
      a) all GPU streams are busy 
      or
      b) we're at the and of the class
      so let's wait for the stream that was scheduled first, or any other busy one */
      timer_init(&timer2);

      if (k_min >= k_max)
      {
        i = count % mystuff->num_streams; // at the end of the class: just wait for the last stream
      }
      else
      {
        i = (count - running) % mystuff->num_streams;  // the oldest still running stream
      }

      if (mystuff->stream_status[i] != RUNNING)     // if that one is not running, take the first running one
      {
        for(i=0; (mystuff->stream_status[i] != RUNNING) && (i<mystuff->num_streams); i++) ;
      }

      if(i<mystuff->num_streams)
      {
#ifdef DEBUG_STREAM_SCHEDULE
        printf(" STREAM_SCHEDULE: Wait for stream %d, already waited %" PRIu64 "us, %d times of %d blocks\n", i, twait, cwait, count);
#endif
        status = clWaitForEvents(1, &mystuff->exec_events[i]); // wait for completion
        if(status != CL_SUCCESS) 
        { 
	        std::cerr<< "Error " << status << ": Waiting for kernel call to finish. (clWaitForEvents)\n";
	      	return RET_ERROR;
        }
      }
      else
      {
#ifdef DEBUG_STREAM_SCHEDULE
        printf(" STREAM_SCHEDULE: Tried to wait but nothing is running!\n");
#endif
        running = 0; /* if nothing is running, correct this if necessary */
      }

#ifdef DEBUG_STREAM_SCHEDULE
      unsigned long long twait1 = timer_diff(&timer2);
      printf(" STREAM_SCHEDULE: Waited %" PRIu64 "us, %d blocks running.\n", twait1, running);
      if (running > 1) twait += twait1;  // don't count the waiting period for the last block as this is unavoidable
#else
      if (running > 1) twait += timer_diff(&timer2); // see above. Note this would not work for num_streams=1, and not reliably for num_streams=2
#endif

      cwait++;
      // technically the stream we've waited for is finished, but
      // leave the stream in status RUNNING to let the case-loop above check for errors and do cleanup
    }
  }

  // all done?

  for(i=0; i<mystuff->num_streams; i++)
  {
    if (mystuff->stream_status[i] != UNUSED)
    { // should not happen
      std::cerr << "Block " << count -i << ", k_min=" << k_min_grid[i] << " in h_ktab[" << i << "] not yet complete!\n";
    }
  }

  status = clEnqueueReadBuffer(QUEUE,
                mystuff->d_RES,
                CL_TRUE,
                0,
                32 * sizeof(int),
                mystuff->h_RES,
                0,
                NULL,
                NULL);
    
  if(status != CL_SUCCESS) 
	{ 
    std::cout << "Error " << status << ": clEnqueueReadBuffer RES failed.\n";
    return RET_ERROR;
  }

#ifdef DETAILED_INFO
  printArray("RES", mystuff->h_RES, 32, 0);
#endif

#ifdef CHECKS_MODBASECASE
  status = clEnqueueReadBuffer(QUEUE,
                mystuff->d_modbasecase_debug,
                CL_TRUE,
                0,
                32 * sizeof(int),
                mystuff->h_modbasecase_debug,
                0,
                NULL,
                NULL);
    
  if(status != CL_SUCCESS) 
	{ 
    std::cout << "Error " << status << ": clEnqueueReadBuffer modbasecase_debug failed.\n";
    return RET_ERROR;
  }

#ifdef DETAILED_INFO
  printArray("modbasecase_debug", mystuff->h_modbasecase_debug, 32, 0);
#endif
  for(i=0;i<32;i++)if(mystuff->h_modbasecase_debug[i] != 0)printf("h_modbasecase_debug[%2d] = %u\n", i, mystuff->h_modbasecase_debug[i]);
#endif

  mystuff->stats.grid_count = count;
  mystuff->stats.class_time = timer_diff(&timer)/1000;
/* prevent division by zero if timer resolution is too low */
  if(mystuff->stats.class_time == 0)mystuff->stats.class_time = 1;
  mystuff->stats.cpu_wait_time = twait;

  if(mystuff->stats.grid_count > 2 * mystuff->num_streams)mystuff->stats.cpu_wait = (float)twait / ((float)mystuff->stats.class_time * 10);
  else                                mystuff->stats.cpu_wait = -1.0f;

  print_status_line(mystuff);

  if(mystuff->stats.cpu_wait >= 0.0f)
  {
/* if SievePrimesAdjust is enable lets try to get 2 % < CPU wait < 6% */
    if(mystuff->sieve_primes_adjust == 1 && mystuff->stats.cpu_wait > 6.0f && mystuff->sieve_primes < mystuff->sieve_primes_upper_limit && (mystuff->mode != MODE_SELFTEST_SHORT))
    {
      mystuff->sieve_primes *= 9;
      mystuff->sieve_primes /= 8;
      if(mystuff->sieve_primes > mystuff->sieve_primes_upper_limit) mystuff->sieve_primes = mystuff->sieve_primes_upper_limit;
    }
    if(mystuff->sieve_primes_adjust == 1 && mystuff->stats.cpu_wait < 2.0f  && mystuff->sieve_primes > mystuff->sieve_primes_min && (mystuff->mode != MODE_SELFTEST_SHORT))
    {
      mystuff->sieve_primes *= 7;
      mystuff->sieve_primes /= 8;
      if(mystuff->sieve_primes < mystuff->sieve_primes_min) mystuff->sieve_primes = mystuff->sieve_primes_min;
    }
  }


  factorsfound=mystuff->h_RES[0];
  for(i=0; (i<factorsfound) && (i<10); i++)
  {
    factor.d2  = mystuff->h_RES[i*3 + 1];
    factor.d1  = mystuff->h_RES[i*3 + 2];
    factor.d0  = mystuff->h_RES[i*3 + 3];
    if ((use_kernel == _71BIT_MUL24) || (use_kernel == _63BIT_MUL24) || (use_kernel == BARRETT70_MUL24))
    {
      print_dez72(factor,string);
    }
    else if (((use_kernel >= BARRETT73_MUL15) && (use_kernel <= BARRETT82_MUL15)) || (use_kernel == MG88))
    {
      print_dez90(factor, string);
    }
    else
    {
      print_dez96(factor, string);
    }
    print_factor(mystuff, i, string);
  }
  if(factorsfound>=10)
  {
    print_factor(mystuff, factorsfound, NULL);
  }

  return factorsfound;
}


/* copy of the init and test functions for troubleshooting and playing around */

void CL_test(cl_int devnumber)
{
  cl_int status;
  size_t dev_s;
  cl_uint numplatforms, i;
  cl_platform_id platform = NULL;
  cl_platform_id* platformlist = NULL;
  cl_device_type devtype = CL_DEVICE_TYPE_GPU;

  if (devnumber < 0)
  {
    devtype = CL_DEVICE_TYPE_CPU;
    devnumber = 0;
  }


  status = clGetPlatformIDs(0, NULL, &numplatforms);
  if(status != CL_SUCCESS)
  {
    std::cerr << "Error " << status << ": clGetPlatformsIDs(num)\n";
  }

  if(numplatforms > 0)
  {
    platformlist = new cl_platform_id[numplatforms];
    status = clGetPlatformIDs(numplatforms, platformlist, NULL);
    if(status != CL_SUCCESS)
    {
      std::cerr << "Error " << status << ": clGetPlatformsIDs\n";
    }

    if (devnumber > 10) // platform number specified as part of -d
    {
      i = devnumber/10 - 1;
      if (i < numplatforms)
      {
        platform = platformlist[i];
        char buf[128];
        status = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR,
                        sizeof(buf), buf, NULL);
        if(status != CL_SUCCESS)
        {
          std::cerr << "Error " << status << ": clGetPlatformInfo(VENDOR)\n";
        }
        std::cout << "OpenCL Platform " << i+1 << "/" << numplatforms << ": " << buf;

        status = clGetPlatformInfo(platform, CL_PLATFORM_VERSION,
                        sizeof(buf), buf, NULL);
        if(status != CL_SUCCESS)
        {
          std::cerr << "Error " << status << ": clGetPlatformInfo(VERSION)\n";
        }
        std::cout << ", Version: " << buf << std::endl;
      }
      else
      {
        fprintf(stderr, "Error: Only %d platforms found. Cannot use platform %d (bad parameter to option -d).\n", numplatforms, i);
      }
    }
    else for(i=0; i < numplatforms; i++) // autoselect: search for AMD
    {
      char buf[128];
      status = clGetPlatformInfo(platformlist[i], CL_PLATFORM_VENDOR,
                        sizeof(buf), buf, NULL);
      if(status != CL_SUCCESS)
      {
        std::cerr << "Error " << status << ": clGetPlatformInfo(VENDOR)\n";
      }
      if(strcmp(buf, "Advanced Micro Devices, Inc.") == 0)
      {
        platform = platformlist[i];
      }
      std::cout << "OpenCL Platform " << i+1 << "/" << numplatforms << ": " << buf;

      status = clGetPlatformInfo(platformlist[i], CL_PLATFORM_VERSION,
                        sizeof(buf), buf, NULL);
      if(status != CL_SUCCESS)
      {
        std::cerr << "Error " << status << ": clGetPlatformInfo(VERSION)\n";
      }
      std::cout << ", Version: " << buf << std::endl;
    }
  }

  delete[] platformlist;
  
  if(platform == NULL)
  {
    std::cerr << "Error: No platform found\n";
  }

  cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
  context = clCreateContextFromType(cps, devtype, NULL, NULL, &status);
  if (status == CL_DEVICE_NOT_FOUND)
  {
    clReleaseContext(context);
    std::cout << "GPU not found, fallback to CPU." << std::endl;
    context = clCreateContextFromType(cps, CL_DEVICE_TYPE_CPU, NULL, NULL, &status);
    if(status != CL_SUCCESS) 
  	{  
   	  std::cerr << "Error " << status << ": clCreateContextFromType(CPU)\n";
    }
  }
  else if(status != CL_SUCCESS) 
	{  
		std::cerr << "Error " << status << ": clCreateContextFromType(GPU)\n";
  }

  cl_uint num_devices;
  status = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(num_devices), &num_devices, NULL);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr << "Error " << status << ": clGetContextInfo(CL_CONTEXT_NUM_DEVICES) - assuming one device\n";
    num_devices = 1;
	}

  status = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &dev_s);
  if(status != CL_SUCCESS) 
	{  
		std::cerr << "Error " << status << ": clGetContextInfo(numdevs)\n";
	}

	if(dev_s == 0)
	{
		std::cerr << "Error: no devices.\n";
	}

  devices = (cl_device_id *)malloc(dev_s*sizeof(cl_device_id));  // *sizeof(...) should not be needed (dev_s is in bytes)
	if(devices == 0)
	{
		std::cerr << "Error: Out of memory.\n";
	}

  status = clGetContextInfo(context, CL_CONTEXT_DEVICES, dev_s*sizeof(cl_device_id), devices, NULL);
  if(status != CL_SUCCESS) 
	{ 
		std::cerr << "Error " << status << ": clGetContextInfo(devices)\n";
	}

  devnumber = devnumber % 10;  // use only the last digit as device number, counting from 1
  cl_uint dev_from=0, dev_to=num_devices;
  if (devnumber > 0)
  {
    if ((cl_uint)devnumber > num_devices)
    {
      fprintf(stderr, "Error: Only %d devices found. Cannot use device %d (bad parameter to option -d).\n", num_devices, devnumber);
    }
    else
    {
      dev_to    = devnumber;    // tweak the loop to run only once for our device
      dev_from  = --devnumber;  // index from 0
    }
  }
   
  for (i=dev_from; i<dev_to; i++)
  {

    status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(deviceinfo.d_name), deviceinfo.d_name, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_NAME)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(deviceinfo.d_ver), deviceinfo.d_ver, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_VERSION)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(deviceinfo.v_name), deviceinfo.v_name, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_VENDOR)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(deviceinfo.dr_version), deviceinfo.dr_version, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DRIVER_VERSION)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(deviceinfo.exts), deviceinfo.exts, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_EXTENSIONS)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(deviceinfo.gl_cache), &deviceinfo.gl_cache, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceinfo.gl_mem), &deviceinfo.gl_mem, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_GLOBAL_MEM_SIZE)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(deviceinfo.max_clock), &deviceinfo.max_clock, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(deviceinfo.units), &deviceinfo.units, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_MAX_COMPUTE_UNITS)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(deviceinfo.wg_size), &deviceinfo.wg_size, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(deviceinfo.w_dim), &deviceinfo.w_dim, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(deviceinfo.wi_sizes), deviceinfo.wi_sizes, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES)\n";
	  }
    status = clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(deviceinfo.l_mem), &deviceinfo.l_mem, NULL);
    if(status != CL_SUCCESS) 
	  { 
		  std::cerr << "Error " << status << ": clGetContextInfo(CL_DEVICE_LOCAL_MEM_SIZE)\n";
	  }
    
    std::cout << "Device " << i+1  << "/" << num_devices << ": " << deviceinfo.d_name << " (" << deviceinfo.v_name << "),\ndevice version: "
      << deviceinfo.d_ver << ", driver version: " << deviceinfo.dr_version << "\nExtensions: " << deviceinfo.exts
      << "\nGlobal memory:" << deviceinfo.gl_mem << ", Global memory cache: " << deviceinfo.gl_cache
      << ", local memory: " << deviceinfo.l_mem << ", workgroup size: " << deviceinfo.wg_size << ", Work dimensions: " << deviceinfo.w_dim
      << "[" << deviceinfo.wi_sizes[0] << ", " << deviceinfo.wi_sizes[1] << ", " << deviceinfo.wi_sizes[2] << ", " << deviceinfo.wi_sizes[3] << ", " << deviceinfo.wi_sizes[4]
      << "] , Max clock speed:" << deviceinfo.max_clock << ", compute units:" << deviceinfo.units << std::endl;
  }

  deviceinfo.maxThreadsPerBlock = deviceinfo.wi_sizes[0];
  deviceinfo.maxThreadsPerGrid  = deviceinfo.wi_sizes[0];
  for (i=1; i<deviceinfo.w_dim && i<5; i++)
  {
    if (deviceinfo.wi_sizes[i])
      deviceinfo.maxThreadsPerGrid *= deviceinfo.wi_sizes[i];
  }

	size_t size;
	char*  source;

	std::fstream f(KERNEL_FILE, (std::fstream::in | std::fstream::binary));

	if(f.is_open())
	{
		f.seekg(0, std::fstream::end);
		size = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);

		source = (char *) malloc(size+1);
		if(!source)
		{
			f.close();
      std::cerr << "\noom\n";
		}

		f.read(source, size);
		f.close();
		source[size] = '\0';
  }
	else
	{
		std::cerr << "\nKernel file \""KERNEL_FILE"\" not found, it needs to be in the same directory as the executable.\n";
	}

  program = clCreateProgramWithSource(context, 1, (const char **)&source, &size, &status);
	if(status != CL_SUCCESS) 
	{ 
	  std::cerr << "Error " << status << ": clCreateProgramWithSource\n";
	}

  status = clBuildProgram(program, 1, &devices[devnumber], "-O3 -I. -DVECTOR_SIZE=4", NULL, NULL);
  { 
      cl_int logstatus;
      char *buildLog = NULL;
      size_t buildLogSize = 0;
      logstatus = clGetProgramBuildInfo (program, devices[devnumber], CL_PROGRAM_BUILD_LOG, 
                buildLogSize, buildLog, &buildLogSize);
      if(logstatus != CL_SUCCESS)
      {
        std::cerr << "Error " << logstatus << ": clGetProgramBuildInfo failed.";
      }
      buildLog = (char*)calloc(buildLogSize,1);
      if(buildLog == NULL)
      {
        std::cerr << "\noom\n";
      }

      logstatus = clGetProgramBuildInfo (program, devices[devnumber], CL_PROGRAM_BUILD_LOG, 
                buildLogSize, buildLog, NULL);
      if(logstatus != CL_SUCCESS)
      {
        std::cerr << "Error " << logstatus << ": clGetProgramBuildInfo failed.";
        free(buildLog);
      }

      std::cout << " \n\tBUILD OUTPUT\n";
      std::cout << buildLog << std::endl;
      std::cout << " \tEND OF BUILD OUTPUT\n";
      free(buildLog);
		std::cerr<<"Error " << status << ": clBuildProgram\n";
  }

  free(source);  

  /* get kernel by name */

  for (i=_TEST_MOD_; i<UNKNOWN_KERNEL; i++)
  {
    printf("."); fflush(stdout);
    kernel_info[i].kernel = clCreateKernel(program, kernel_info[i].kernelname, &status);
    if(status != CL_SUCCESS) 
  	{  
  		std::cerr<<"Error " << status << ": Creating Kernel " << kernel_info[i].kernelname << " from program. (clCreateKernel)\n";
  	}
  }

  /* init_streams */

  mystuff.threads_per_grid = 1024 * 1024;
  
  if (context==NULL)
  {
    fprintf(stderr, "invalid context.\n");
  }

  for(i=0;i<(mystuff.num_streams);i++)
  {
    mystuff.stream_status[i] = UNUSED;
    if( (mystuff.h_ktab[i] = (cl_uint *) malloc( mystuff.threads_per_grid * sizeof(cl_uint))) == NULL )
    {
      printf("ERROR: malloc(h_ktab[%d]) failed\n", i);
    }
    mystuff.d_ktab[i] = clCreateBuffer(context, 
                      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                      mystuff.threads_per_grid * sizeof(cl_uint),
                      mystuff.h_ktab[i], 
                      &status);
    if(status != CL_SUCCESS) 
  	{ 
	  	std::cout<<"Error " << status << ": clCreateBuffer (h_ktab[" << i << "]) \n";
	  }
  }
  if( (mystuff.h_RES = (cl_uint *) malloc(32 * sizeof(cl_uint))) == NULL )
  {
    printf("ERROR: malloc(h_RES) failed\n");
  }
  mystuff.d_RES = clCreateBuffer(context,
                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    32 * sizeof(cl_uint),
                    mystuff.h_RES,
                    &status);
  if(status != CL_SUCCESS) 
  { 
		std::cout<<"Error " << status << ": clCreateBuffer (d_RES)\n";
	}


  // Now, quickly test one kernel ...
  // (10 * 2^64+25) mod 3 * 2^23
  long long unsigned int hi=10;
  long long unsigned int lo=25;
  long long unsigned int q=3<<23;
  cl_float qr=0.9998f/(cl_float)q;
  long long unsigned int res_hi;
  long long unsigned int res_lo;

  cl_event in_evt, mod_evt, res_evt;

  res_hi = res_lo = 0;

  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    0, 
                    sizeof(cl_ulong), 
                    (void *)&hi);
  if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error " << status << ": Setting kernel argument. (hi)\n";
	}
  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    1, 
                    sizeof(cl_ulong), 
                    (void *)&lo);
  if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error " << status << ": Setting kernel argument. (lo)\n";
	}
  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    2, 
                    sizeof(cl_ulong), 
                    (void *)&q);
  if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error " << status << ": Setting kernel argument. (q)\n";
	}
  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    3, 
                    sizeof(cl_float), 
                    (void *)&qr);
  if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error " << status << ": Setting kernel argument. (qr)\n";
	}
  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    4, 
                    sizeof(cl_mem), 
                    (void *)&mystuff.d_RES);
  if(status != CL_SUCCESS) 
	{ 
		std::cout<<"Error " << status << ": Setting kernel argument. (RES)\n";
	}
  // dummy arg if KERNEL_TRACE is enabled: ignore errors if not.
  status = clSetKernelArg(kernel_info[_TEST_MOD_].kernel, 
                    5, 
                    sizeof(cl_uint), 
                    (void *)&status);

  struct timeval timer;
  cl_ulong startTime, endTime;
  size_t g_size[2];

  timer_init(&timer);
#define TEST_LOOPS 10

  for (i=1; i<=TEST_LOOPS; i++)
  {
    printf("loop %d: \n", i);fflush(NULL);

    /* set result array to 0 */ 
    memset(mystuff.h_RES,0,32 * sizeof(int));
    status = clEnqueueWriteBuffer(commandQueuePrf,
                  mystuff.d_RES,
                  CL_FALSE,          // Don't wait for completion; it's fast to copy 128 bytes ;-)
                  0,
                  32 * sizeof(int),
                  mystuff.h_RES,
                  0,
                  NULL,
                  &in_evt);
    if(status != CL_SUCCESS) 
    {  
      std::cout<<"Error " << status << ": Copying h_RES(clEnqueueWriteBuffer)\n";
    }


    if (i<256)
    {
      g_size[0] = (i%256);
      g_size[1] = (i/256)+1;
    }
    else
    {
      g_size[0] = 256;
      g_size[1] = i-255;
    }

    status = clEnqueueNDRangeKernel(commandQueuePrf,
                 kernel_info[_TEST_MOD_].kernel,
                 2,
                 NULL,
                 g_size,
                 NULL,
                 1,
                 &in_evt,
                 &mod_evt);
    if(status != CL_SUCCESS) 
	  { 
	  	std::cerr<< "Error " << status << ": Enqueuing kernel(clEnqueueNDRangeKernel)\n";
	  }

    /*
    try {

    status = clWaitForEvents(1, &mod_evt); 
    } catch(...) {
      std::cerr << "Exception in clWaitForEvents\n";
    }
    if(status != CL_SUCCESS) 
    { 
  	  std::cerr<< "Error " << status << ": Waiting for mod call to finish. (clWaitForEvents)\n";
    }
    */

    status = clEnqueueReadBuffer(commandQueuePrf,
                mystuff.d_RES,
                CL_FALSE,
                0,
                32 * sizeof(int),
                mystuff.h_RES,
                1,
                &mod_evt,
                &res_evt);
    
    if(status != CL_SUCCESS) 
    { 
      std::cerr << "Error " << status << ": clEnqueueReadBuffer RES failed. (clEnqueueReadBuffer)\n";
    }

    try
    {
      status = clFinish(commandQueuePrf);
    }
    catch(...)
    {
	  	std::cerr<< "Exception in clFinish\n";
    }

    if(status != CL_SUCCESS) 
    { 
	  	std::cerr<< "Error " << status << ": clFinish\n";
    }

    /* Get kernel profiling info */
    status = clGetEventProfilingInfo(mod_evt,
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &startTime,
                                0);
     if(status != CL_SUCCESS)
 	   { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
              }
              status = clGetEventProfilingInfo(mod_evt,
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endTime,
                                0);
              if(status != CL_SUCCESS)
 	            { 
		            std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
              }
 //             std::cout<< "mod_kernel finished in " << (endTime - startTime)/1e3 << " us.\n" ;

    status = clReleaseEvent(in_evt);
    if(status != CL_SUCCESS) 
    { 
	  	std::cerr<< "Error " << status << ": Release in event object. (clReleaseEvent)\n";
    }

    status = clReleaseEvent(mod_evt);
    if(status != CL_SUCCESS) 
    { 
	  	std::cerr<< "Error " << status << ": Release mod event object. (clReleaseEvent)\n";
    }

    status = clReleaseEvent(res_evt);
    if(status != CL_SUCCESS) 
    { 
	  	std::cerr<< "Error " << status << ": Release res event object. (clReleaseEvent)\n";
    }


//    std::cout << "Avg. test kernel runtime (incl. overhead): " << timer_diff(&timer)/TEST_LOOPS << " us.\n";


    printf("%d threads: ", (int)(g_size[0]*g_size[1]));
    printArray("RES", mystuff.h_RES, 32, 0);

  }
}
