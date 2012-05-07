/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2012  Oliver Weihe (o.weihe@t-online.de)
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

#include <stdio.h>
#include <stdlib.h>
#ifndef _MSC_VER
#include <unistd.h>
#endif
#include <string.h>
#include <errno.h>
#include <time.h>
#include <CL/cl.h>

#include "params.h"
#include "my_types.h"
#include "compatibility.h"
#include "sieve.h"
#include "read_config.h"
#include "parse.h"
#include "timer.h"
#include "checkpoint.h"
#include "signal_handler.h"
#include "filelocking.h"

#include "mfakto.h"


mystuff_t mystuff;

extern OpenCL_deviceinfo_t deviceinfo;
extern kernel_info_t       kernel_info[];

unsigned long long int calculate_k(unsigned int exp, int bits)
/* calculates biggest possible k in "2 * exp * k + 1 < 2^bits" */
{
  unsigned long long int k = 0, tmp_low, tmp_hi;
  
  if((bits > 65) && exp < (unsigned int)(1U << (bits - 65))) k = 0; // k would be >= 2^64...
  else if(bits <= 64)
  {
    tmp_low = 1ULL << (bits - 1);
    tmp_low--;
    k = tmp_low / exp;
  }
  else if(bits <= 96)
  {
    tmp_hi = 1ULL << (bits - 33);
    tmp_hi--;
    tmp_low = 0xFFFFFFFFULL;
    
    k = tmp_hi / exp;
    tmp_low += (tmp_hi % exp) << 32;
    k <<= 32;
    k += tmp_low / exp;
  }
  
  if(k == 0)k = 1;
  return k;
}




int tf(unsigned int exp, int bit_min, int bit_max, mystuff_t *mystuff, int class_hint, unsigned long long int k_hint, enum GPUKernels use_kernel)
/*
tf M<exp> from 2^bit_min to 2^bit_max

kernel: see my_types.h -> enum GPUKernels

return value (mystuff->mode = MODE_NORMAL):
number of factors found
RET_ERROR any CL function returned an error
RET_QUIT if early exit was requested by SIGINT

 

return value (mystuff->mode > MODE_NORMAL), i.e. selftest:
0 for a successful selftest (known factor was found)
1 no factor found
2 wrong factor returned
RET_ERROR any CL function returned an error

other return value 
-1 unknown mode
*/
{
  unsigned int cur_class, max_class = NUM_CLASSES-1, i, count = 0;
  unsigned long long int k_min, k_max, k_range, tmp;
  unsigned int f_hi, f_med, f_low;
  struct timeval timer;
  time_t time_last_checkpoint;
#ifdef VERBOSE_TIMING  
  struct timeval timer2;
#endif  
  int factorsfound = 0, numfactors = 0, restart = 0, do_checkpoint = mystuff->checkpoints;
  FILE *resultfile=NULL;

  const char *kernelname;
  int retval = 0;
    
  unsigned long long int time_run, time_est;


  if(mystuff->mode != MODE_SELFTEST_SHORT)printf("Starting trial factoring M%u from 2^%d to 2^%d (%5.2fGHz-days)\n",
    exp, bit_min, bit_max, 0.016968 * (double)(1ULL << (bit_min - 47)) * 1680 / exp * ((1 << (bit_max-bit_min)) -1));
  timer_init(&timer);
  time(&time_last_checkpoint);

  mystuff->class_counter = 0;
  
  k_min=calculate_k(exp,bit_min);
  k_max=calculate_k(exp,bit_max);
  
  if(mystuff->mode > MODE_NORMAL) // any selftest mode
  {
/* a shortcut for the selftest, bring k_min and k_max "close" to the known factor */
    if(NUM_CLASSES == 420)k_range = 10000000000ULL;
    else                  k_range = 100000000000ULL;
    if(mystuff->mode == MODE_SELFTEST_SHORT)k_range /= 5; /* even smaller ranges for the "small" selftest */
    if((k_max - k_min) > (3ULL * k_range))
    {
      tmp = k_hint - (k_hint % k_range) - k_range;
      if(tmp > k_min) k_min = tmp;

      tmp += 3ULL * k_range;
      if((tmp < k_max) || (k_max < k_min)) k_max = tmp; /* check for k_max < k_min enables some selftests where k_max >= 2^64 but the known factor itself has a k < 2^64 */
    }
  }

  k_min -= k_min % NUM_CLASSES;	/* k_min is now 0 mod NUM_CLASSES */

  if(mystuff->mode != MODE_SELFTEST_SHORT)
  {
    printf("  k_min = %llu - k_max = %llu\n", k_min, k_max);
  }

  if(use_kernel == AUTOSELECT_KERNEL)
  {  // this is probably the speed order for VLIW4, Cayman (HD6970), and most likely also for GCN
    if (mystuff->preferredKernel == _71BIT_MUL24)  // maybe this can be bound to some version/feature/capability or be tested during selftest
    {
      if      ((bit_min >= 60) && (bit_max <= 73) && (bit_max - bit_min == 1))  use_kernel = BARRETT73_MUL15;  // 295M/s on HD6970
      else if ((bit_min >= 61) && (bit_max <= 72))                              use_kernel = _71BIT_MUL24;     // 203M/s
      else if ((bit_min >= 63) && (bit_max <= 70) && (bit_max - bit_min == 1))  use_kernel = BARRETT72_MUL24;  // 207M/s
      else if                     (bit_max <= 64)                               use_kernel = _63BIT_MUL24;     // 211M/s
      else if ((bit_min >= 64) && (bit_max <= 79))                              use_kernel = BARRETT79_MUL32;  // 195M/s
      else if ((bit_min >= 64) && (bit_max <= 92) && (bit_max - bit_min == 1))  use_kernel = BARRETT92_MUL32;  // 155M/s
//      else if                     (bit_max <  95)                               use_kernel = _95BIT_64_OpenCL;
    }
    else
    {  // this is the speed order for VLIW5, HD5770, for instance
      if      ((bit_min >= 63) && (bit_max <= 70) && (bit_max - bit_min == 1))  use_kernel = BARRETT72_MUL24;  // 321M/s on HD5870
      else if ((bit_min >= 60) && (bit_max <= 73) && (bit_max - bit_min == 1))  use_kernel = BARRETT73_MUL15;  // 288M/s
      else if ((bit_min >= 61) && (bit_max <= 72))                              use_kernel = _71BIT_MUL24;     // 258M/s
      else if ((bit_min >= 64) && (bit_max <= 79))                              use_kernel = BARRETT79_MUL32;  // 255M/s
      else if                     (bit_max <= 64)                               use_kernel = _63BIT_MUL24;     // 236M/s
      else if ((bit_min >= 64) && (bit_max <= 92) && (bit_max - bit_min == 1))  use_kernel = BARRETT92_MUL32;  // 205M/s
//      else if                     (bit_max <  95)                               use_kernel = _95BIT_64_OpenCL;
    }

    if(use_kernel == AUTOSELECT_KERNEL)
    {
      printf("ERROR: No suitable kernel found for bit_min=%d, bit_max=%d.\n",
                 bit_min, bit_max);
      return RET_ERROR;
    }
  }

  kernelname=kernel_info[use_kernel].kernelname;

  if(mystuff->mode != MODE_SELFTEST_SHORT)printf("Using GPU kernel \"%s\"\n",kernelname);

  if(mystuff->mode == MODE_NORMAL)
  {
    if((mystuff->checkpoints > 0) && (checkpoint_read(exp, bit_min, bit_max, &cur_class, &factorsfound) == 1))
    {
      printf("\nfound a valid checkpoint file!\n");
      printf("  last finished class was: %d\n", cur_class);
      printf("  found %d factor%s already\n\n", factorsfound, factorsfound == 1 ? "" : "s");
      cur_class++; // the checkpoint contains the last completely processed class!

/* calculate the number of classes which are already processed. This value is needed to estimate ETA */
      for(i = 0; i < cur_class; i++)
      {
/* check if class is NOT "3 or 5 mod 8", "0 mod 3", "0 mod 5", "0 mod 7" (or "0 mod 11") */
        if( ((2 * (exp% 8) * ((k_min+i)% 8)) % 8 !=  2) &&
            ((2 * (exp% 8) * ((k_min+i)% 8)) % 8 !=  4) &&
            ((2 * (exp% 3) * ((k_min+i)% 3)) % 3 !=  2) &&
            ((2 * (exp% 5) * ((k_min+i)% 5)) % 5 !=  4) &&
#ifdef MORE_CLASSES
            ((2 * (exp%11) * ((k_min+i)%11)) %11 != 10) &&
#endif
            ((2 * (exp% 7) * ((k_min+i)% 7)) % 7 !=  6))
        {
          mystuff->class_counter++;
        }
      }
      restart = mystuff->class_counter;
    }
    else
    {
      cur_class=0;
    }
  }
  else // mystuff->mode != MODE_NORMAL
  {
    cur_class = class_hint % NUM_CLASSES;
    max_class = cur_class;
  }

  for(; cur_class <= max_class; cur_class++)
  {
/* check if class is NOT "3 or 5 mod 8", "0 mod 3", "0 mod 5", "0 mod 7" (or "0 mod 11") */
    if( ((2 * (exp% 3) * ((k_min+cur_class)% 3)) % 3 !=  2) &&
        ((2 * (exp% 5) * ((k_min+cur_class)% 5)) % 5 !=  4) &&
        ((2 * (exp% 8) * ((k_min+cur_class)% 8)) % 8 !=  2) &&
        ((2 * (exp% 8) * ((k_min+cur_class)% 8)) % 8 !=  4) &&
#ifdef MORE_CLASSES        
        ((2 * (exp%11) * ((k_min+cur_class)%11)) %11 != 10) &&
#endif    
        ((2 * (exp% 7) * ((k_min+cur_class)% 7)) % 7 !=  6))
    {
      if(mystuff->quit)
      {
/* check if quit is requested. Because this is at the begining of the class
   we can be sure that if RET_QUIT is returned the last class hasn't
   finished. The signal handler which sets mystuff->quit not active during
   selftests so we need to check for RET_QUIT only when doing real work. */
        if(mystuff->printmode == 1)printf("\n");
        return RET_QUIT;
      }
      else
      {

#ifdef VERBOSE_TIMING
        timer_init(&timer2);
#endif    
        if (mystuff->sieve_gpu == 1)
        {
          cl_ulong new_k_min=k_min+cur_class;
          run_cl_sieve_init(exp, k_min+cur_class, 256);
//          run_cl_sieve_init(exp, k_min+cur_class, mystuff->sieve_primes);
          run_cl_sieve(exp, &new_k_min, 256);
//          run_cl_sieve(exp, &new_k_min, mystuff->sieve_primes);
        }
        else
        {
          sieve_init_class(exp, k_min+cur_class, mystuff->sieve_primes);
        }
#ifdef VERBOSE_TIMING      
        printf("tf(): time spent for sieve_init_class(exp, k_min+cur_class, mystuff->sieve_primes): %" PRIu64 "ms\n",timer_diff(&timer2)/1000);
#endif
        if(mystuff->mode != MODE_SELFTEST_SHORT && (count == 0 || (count%20 == 0 && mystuff->printmode == 0)))
        {
          printf("%s\n", mystuff->head_line);
        }
        count++;
        mystuff->class_counter++;
        if (mystuff->p_par[CLASS_NUM].pos) sprintf(mystuff->p_par[CLASS_NUM].out, "%3d", mystuff->class_counter);
        if (mystuff->p_par[PCT_COMPLETE].pos) sprintf(mystuff->p_par[PCT_COMPLETE].out, "%6.2f", 0.1041666667f * mystuff->class_counter);

        switch (use_kernel)
        {
          case _71BIT_MUL24:
          case _63BIT_MUL24:
          case _64BIT_64_OpenCL:
          case _95BIT_64_OpenCL:
          case BARRETT58_MUL15:
          case BARRETT73_MUL15:
          case BARRETT72_MUL24:
          case BARRETT79_MUL32:
          case BARRETT92_MUL32:
          case BARRETT92_64_OpenCL: numfactors = tf_class_opencl (exp, bit_min, bit_max, k_min+cur_class, k_max, mystuff, use_kernel); break;
          default:  printf("ERROR: Unknown kernel selected (%d)!\n", use_kernel);  return RET_ERROR;
        }

        if (numfactors == RET_ERROR)
        {
          printf("ERROR from tf_class.\n");
          return RET_ERROR;
        }
        factorsfound+=numfactors;

        if(mystuff->mode == MODE_NORMAL)
        {
          if (mystuff->checkpoints > 0)
          {
            if ( ((mystuff->checkpoints > 1) && (--do_checkpoint == 0)) ||
                 ((mystuff->checkpoints == 1) && (time(NULL) - time_last_checkpoint > (time_t) mystuff->checkpointdelay)) ||
                   mystuff->quit )
            {
              checkpoint_write(exp, bit_min, bit_max, cur_class, factorsfound);
              do_checkpoint = mystuff->checkpoints;
              time(&time_last_checkpoint);
            }
          }
          if((mystuff->stopafterfactor >= 2) && (factorsfound > 0) && (cur_class != max_class))cur_class = max_class + 1;
        }
      }
      fflush(NULL);
    }
  }
  if(mystuff->mode != MODE_SELFTEST_SHORT && mystuff->printmode == 1)printf("\n");
  if(mystuff->mode == MODE_NORMAL)resultfile=fopen_and_lock(mystuff->resultsfile, "a");
  
  if(factorsfound)
  {
#ifndef MORE_CLASSES
    if((mystuff->mode == MODE_NORMAL) && (mystuff->class_counter < 96))
#else
    if((mystuff->mode == MODE_NORMAL) && (mystuff->class_counter < 960))
#endif
    {
      fprintf(resultfile, "found %d factor%s for M%u from 2^%2d to 2^%2d (partially tested) [%s %s_%d]\n",
        factorsfound, (factorsfound > 1) ? "s" : "", exp, bit_min, bit_max, MFAKTO_VERSION, kernelname, mystuff->vectorsize);
      printf(             "found %d factor%s for M%u from 2^%2d to 2^%2d (partially tested) [%s %s_%d]\n",
        factorsfound, (factorsfound > 1) ? "s" : "", exp, bit_min, bit_max, MFAKTO_VERSION, kernelname, mystuff->vectorsize);
    }
    else
    {
      if(mystuff->mode == MODE_NORMAL)        fprintf(resultfile, "found %d factor%s for M%u from 2^%2d to 2^%2d [%s %s_%d]\n",
        factorsfound, (factorsfound > 1) ? "s" : "", exp, bit_min, bit_max, MFAKTO_VERSION, kernelname, mystuff->vectorsize);
      if(mystuff->mode != MODE_SELFTEST_SHORT)printf(             "found %d factor%s for M%u from 2^%2d to 2^%2d [%s %s_%d]\n",
        factorsfound, (factorsfound > 1) ? "s" : "", exp, bit_min, bit_max, MFAKTO_VERSION, kernelname, mystuff->vectorsize);
    }
  }
  else
  {
    if(mystuff->mode == MODE_NORMAL)        fprintf(resultfile, "no factor for M%u from 2^%d to 2^%d [%s %s_%d]\n",
      exp, bit_min, bit_max, MFAKTO_VERSION, kernelname, mystuff->vectorsize);
    if(mystuff->mode != MODE_SELFTEST_SHORT)printf(             "no factor for M%u from 2^%d to 2^%d [%s %s_%d]\n",
      exp, bit_min, bit_max, MFAKTO_VERSION, kernelname, mystuff->vectorsize);
  }
  if(mystuff->mode == MODE_NORMAL)
  {
    retval = factorsfound;
    unlock_and_fclose(resultfile);
    if(mystuff->checkpoints > 0)checkpoint_delete(exp);
  }
  else // mystuff->mode != MODE_NORMAL
  {
    if(mystuff->h_RES[0] == 0)
    {
      printf("ERROR: selftest failed for M%u (%s)\n", exp, kernel_info[use_kernel].kernelname);
      printf("  no factor found\n");
      retval = 1;
    }
    else // mystuff->h_RES[0] > 0
    {
/*
calculate the value of the known factor in f_{hi|med|low} and compare with the
results from the selftest.
k_max and k_min are used as 64bit temporary integers here...
*/    
      f_hi    = (k_hint >> 63);
      f_med   = (k_hint >> 31) & 0xFFFFFFFFULL;
      f_low   = (k_hint <<  1) & 0xFFFFFFFFULL; /* f_{hi|med|low} = 2 * k_hint */
      
      k_max   = (unsigned long long int)exp * f_low;
      f_low   = (k_max & 0xFFFFFFFFULL) + 1;
      k_min   = (k_max >> 32);

      k_max   = (unsigned long long int)exp * f_med;
      k_min  += k_max & 0xFFFFFFFFULL;
      f_med   = k_min & 0xFFFFFFFFULL;
      k_min >>= 32;
      k_min  += (k_max >> 32);

      f_hi  = (unsigned int ) (k_min + (exp * f_hi)); /* f_{hi|med|low} = 2 * k_hint * exp +1 */
      
      if ((use_kernel == _71BIT_MUL24) || (use_kernel == _63BIT_MUL24) || (use_kernel == BARRETT72_MUL24)) /* 71bit kernel uses only 24bit per int */
      {
        f_hi  <<= 16;
        f_hi   += f_med >> 16;

        f_med <<= 8;
        f_med  += f_low >> 24;
        f_med  &= 0x00FFFFFF;
        
        f_low  &= 0x00FFFFFF;
      }
      else if ((use_kernel == BARRETT73_MUL15) || (use_kernel == BARRETT58_MUL15))
      {
        // 30 bits per reported result int
        f_hi  <<= 4;
        f_hi   += f_med >> 28;

        f_med <<= 2;
        f_med  += f_low >> 30;
        f_med  &= 0x3FFFFFFF;
        
        f_low  &= 0x3FFFFFFF;
      }
      k_min=0; /* using k_min for counting the number of matches here */
      for(i=0; (i<mystuff->h_RES[0]) && (i<10); i++)
      {
        if(mystuff->h_RES[i*3 + 1] == f_hi  && \
           mystuff->h_RES[i*3 + 2] == f_med && \
           mystuff->h_RES[i*3 + 3] == f_low) k_min++;
      }
      if(k_min != 1) /* the factor should appear ONCE */
      {
        printf("ERROR: selftest failed for M%u (%s)\n", exp, kernel_info[use_kernel].kernelname);
        printf("  expected result: %08X %08X %08X\n", f_hi, f_med, f_low);
        for(i=0; (i<mystuff->h_RES[0]) && (i<10); i++)
        {
          printf("  reported result: %08X %08X %08X\n", mystuff->h_RES[i*3 + 1], mystuff->h_RES[i*3 + 2], mystuff->h_RES[i*3 + 3]);
        }
        retval = 2;
      }
      else
      {
        if(mystuff->mode != MODE_SELFTEST_SHORT)printf("selftest for M%u passed (%s)!\n", exp, kernel_info[use_kernel].kernelname);
      }
    }
  }
  if(mystuff->mode != MODE_SELFTEST_SHORT)
  {
    time_run = timer_diff(&timer)/1000;
    
    if(restart == 0)printf("tf(): total time spent: ");
    else            printf("tf(): time spent since restart:   ");

/*  restart == 0 ==> time_est = time_run */

    if(time_run > 86400000ULL)printf("%" PRIu64 "d ",   time_run / 86400000ULL);
    if(time_run > 3600000ULL) printf("%2" PRIu64 "h ", (time_run /  3600000ULL) % 24ULL);
    if(time_run > 60000ULL)   printf("%2" PRIu64 "m ", (time_run /    60000ULL) % 60ULL);
                              printf("%2" PRIu64 ".%03" PRIu64 "s\n", (time_run / 1000ULL) % 60ULL, time_run % 1000ULL);
    if(restart != 0)
    {
      time_est = (time_run * mystuff->class_counter ) / (unsigned long long int)(mystuff->class_counter-restart);
      printf("      estimated total time spent: ");
      if(time_est > 86400000ULL)printf("%" PRIu64 "d ",   time_est / 86400000ULL);
      if(time_est > 3600000ULL) printf("%2" PRIu64 "h ", (time_est /  3600000ULL) % 24ULL);
      if(time_est > 60000ULL)   printf("%2" PRIu64 "m ", (time_est /    60000ULL) % 60ULL);
                                printf("%2" PRIu64 ".%03" PRIu64 "s\n", (time_est / 1000ULL) % 60ULL, time_est % 1000ULL);
    }
    printf("\n");
  }
  return retval;
}


void print_help(char *string)
{
  printf("mfaktc (%s) Copyright (C) 2009-2011  Oliver Weihe (o.weihe@t-online.de),\n", MFAKTO_VERSION);
  printf("                                                 Bertram Franz (bertramf@gmx.net)\n");
  printf("This program comes with ABSOLUTELY NO WARRANTY; for details see COPYING.\n");
  printf("This is free software, and you are welcome to redistribute it\n");
  printf("under certain conditions; see COPYING for details.\n\n\n");

  printf("Usage: %s [options]\n", string);
  printf("  -h|--help              display this help and exit\n");
  printf("  -d <xy>                specify to use OpenCL platform number x and\n");
  printf("                         device number y in this program\n");
  printf("  -d c                   force using all CPUs\n");
  printf("  -d g                   force using the first GPU\n");
  printf("  -tf <exp> <min> <max>  trial factor M<exp> from 2^<min> to 2^<max> and exit\n");
  printf("                         instead of parsing the worktodo file\n");
  printf("  -i|--inifile <file>    load <file> as inifile (default: mfakto.ini)\n");
  printf("  -st                    run builtin selftest (half the testcases) and exit\n");
  printf("  -st2                   run builtin selftest (all testcases) and exit\n");
  printf("\n");
  printf("options for debugging purposes\n");
  printf("  --timertest            run test of timer functions and exit\n");
  printf("  --sleeptest            run test of sleep functions and exit\n");
  printf("  --perftest             run performance test of the sieve and other parts, then exit\n");
  printf("  --CLtest               run test of some OpenCL functions and exit\n");
  printf("                         specify -d before --CLtest to test the specified device\n");
}


int selftest(mystuff_t *mystuff, enum MODES type)
/*
type = 1: small selftest (this is executed EACH time mfakto is started)
type = 2: half selftest
type = 3: full selftest

return value
0 selftest passed
1 selftest failed
RET_ERROR we might have a serios problem
*/
{
  int i, j, tf_res, st_success=0, st_nofactor=0, st_wrongfactor=0, st_unknown=0;

#define NUM_SELFTESTS 2597
  unsigned int exp[NUM_SELFTESTS], num_selftests=0;
  int bit_min[NUM_SELFTESTS], f_class, selftests_to_run;
  unsigned long long int k[NUM_SELFTESTS];
  int retval=1, ind;
  enum GPUKernels kernels[9];
  unsigned int index[] = {    2,   25,   39,   57,   // some factors below 2^71 (test the 71/75 bit kernel depending on compute capability)
                             70,   72,   73,  82,  88,   // some factors below 2^75 (test 75 bit kernel)
                            106,  355,  358,  666,   // some very small factors
                           1547, 1552, 1556, 1557    // some factors below 2^95 (test 95 bit kernel)
                         };                          // mfakto special case (25-bit factor)
  if (type == MODE_SELFTEST_FULL)
    selftests_to_run = NUM_SELFTESTS;
  else
    selftests_to_run = 1559;

#include "selftest-data.h"

  if (mystuff->p_par[SIEVE_PRIMES].pos) sprintf(mystuff->p_par[SIEVE_PRIMES].out, "%7d", mystuff->sieve_primes);
  register_signal_handler(mystuff);

  for(i=0; i<selftests_to_run; i++)
  {
    if(type == MODE_SELFTEST_SHORT)
    {
      if (i < (sizeof(index)/sizeof(index[0])))
      {
        ind = index[i];
        printf("########## testcase %d/%d (#%d) ##########\r", i+1, (int) (sizeof(index)/sizeof(index[0])), ind);
      }
      else
        break; // short test done
    }
    else // treat type <> 1 as full test
    {
      printf("########## testcase %d/%d ##########\n", i+1, selftests_to_run);
      ind = i;
    }
    f_class = (int)(k[ind] % NUM_CLASSES);


/* create a list which kernels can handle this testcase */
    j = 0;
    if (bit_min[ind] <= 63)                            kernels[j++] = _63BIT_MUL24;
    if ((bit_min[ind] >= 61) && (bit_min[ind] < 72))   kernels[j++] = _71BIT_MUL24;
    if ((bit_min[ind] >= 64) && (bit_min[ind] < 79))   kernels[j++] = BARRETT79_MUL32; 
    if ((bit_min[ind] >= 64) && (bit_min[ind] < 92))   kernels[j++] = BARRETT92_MUL32; /* no need to check bit_max - bit_min == 1 ;) */
    if ((bit_min[ind] >= 63) && (bit_min[ind] < 70))   kernels[j++] = BARRETT72_MUL24;
    if ((bit_min[ind] >= 60) && (bit_min[ind] < 73))   kernels[j++] = BARRETT73_MUL15;
//
//      if ((bit_min[ind] >= 64) && (bit_min[ind]) < 79)   kernels[j++] = _95BIT_64_OpenCL; // currently just a test for no sieving at all

    if (mystuff->p_par[EXP].pos)         sprintf(mystuff->p_par[EXP].out, "%d", exp[ind]);
    if (mystuff->p_par[LOWER_LIMIT].pos) sprintf(mystuff->p_par[LOWER_LIMIT].out, "%2d", bit_min[ind]);
    if (mystuff->p_par[UPPER_LIMIT].pos) sprintf(mystuff->p_par[UPPER_LIMIT].out, "%2d", bit_min[ind]+1);

    while(j>0)
    {
      num_selftests++;
      tf_res=tf(exp[ind], bit_min[ind], bit_min[ind]+1, mystuff, f_class, k[ind], kernels[--j]);
            if(tf_res == 0)st_success++;
      else if(tf_res == 1)st_nofactor++;
      else if(tf_res == 2)st_wrongfactor++;
      else if(tf_res == RET_ERROR) return RET_ERROR; /* bail out, we might have a serios problem */
      else           st_unknown++;
#ifdef DETAILED_INFO
      printf("Test %d finished, so far suc: %d, no: %d, wr: %d, unk: %d\n", num_selftests, st_success, st_nofactor, st_wrongfactor, st_unknown);
      fflush(NULL);
#endif
      if (mystuff->quit) break;
    }
    if (mystuff->quit) break;
  }

  printf("Selftest statistics                          \n");
  printf("  number of tests           %d\n", num_selftests);
  printf("  successful tests          %d\n", st_success);
  if(st_nofactor > 0)   printf("  no factor found           %d\n", st_nofactor);
  if(st_wrongfactor > 0)printf("  wrong factor reported     %d\n", st_wrongfactor);
  if(st_unknown > 0)    printf("  unknown return value      %d\n", st_unknown);
  printf("\n");

  if(st_success == num_selftests)
  {
    printf("selftest PASSED!\n\n");
    retval=0;
  }
  else
  {
    printf("selftest FAILED!\n\n");
  }

  return retval;
}

int main(int argc, char **argv)
{
  unsigned int exp = 0;
  int bit_min = -1, bit_max = -1, bit_min_stage, bit_max_stage;
  int parse_ret = -1;
  int devicenumber = 0;
#ifdef VERBOSE_TIMING  
  struct timeval timer;
#endif
  int i = 1, tmp = 0;
  char *ptr;
  int use_worktodo = 1;
  
  mystuff.mode=MODE_NORMAL;
  mystuff.quit = 0;
  strcpy(mystuff.inifile, "mfakto.ini");

  while(i<argc)
  {
    if((!strcmp((char*)"-h", argv[i])) || (!strcmp((char*)"--help", argv[i])))
    {
      print_help(argv[0]);
      return 0;
    }
    else if(!strcmp((char*)"-d", argv[i]))
    {
      if(i+1 >= argc)
      {
        printf("ERROR: no device number specified for option \"-d\"\n");
        return 1;
      }
      if (argv[i+1][0] == 'c')  // run on CPU
      {
        devicenumber = -1;
      }
      else if (argv[i+1][0] == 'g')  // run on GPU
      {
        devicenumber = 10000;
      }
      else
      {
        devicenumber = strtol(argv[i+1],&ptr,10);
        if(*ptr || errno || devicenumber != strtol(argv[i+1],&ptr,10) )
        {
          printf("ERROR: can't parse <device number> for option \"-d\"\n");
          return 1;
	      }
      }
      i++;
    }
    else if(!strcmp((char*)"-tf", argv[i]))
    {
      if(i+3 >= argc)
      {
        printf("ERROR: missing parameters for option \"-tf\"\n");
        return 1;
      }
      exp=(unsigned int)strtoul(argv[i+1],&ptr,10);
      if(*ptr || errno || (unsigned long)exp != strtoul(argv[i+1],&ptr,10) )
      {
        printf("ERROR: can't parse parameter <exp> for option \"-tf\"\n");
        return 1;
      }
      bit_min=(int)strtol(argv[i+2],&ptr,10);
      if(*ptr || errno || (long)bit_min != strtol(argv[i+2],&ptr,10) )
      {
        printf("ERROR: can't parse parameter <min> for option \"-tf\"\n");
        return 1;
      }
      bit_max=(int)strtol(argv[i+3],&ptr,10);
      if(*ptr || errno || (long)bit_max != strtol(argv[i+3],&ptr,10) )
      {
        printf("ERROR: can't parse parameter <max> for option \"-tf\"\n");
        return 1;
      }
      if(!valid_assignment(exp, bit_min, bit_max))
      {
        return 1;
      }
      use_worktodo = 0;
      parse_ret = 0;
      i += 3;
    }
    else if(!strcmp((char*)"-st", argv[i]))
    {
      mystuff.mode = MODE_SELFTEST_HALF;
    }
    else if(!strcmp((char*)"-st2", argv[i]))
    {
      mystuff.mode = MODE_SELFTEST_FULL;
    }
    else if(!strcmp((char*)"-i", argv[i]) || !strcmp((char*)"--inifile", argv[i]))
    {
      i++;
      strncpy(mystuff.inifile, argv[i], 50);
      mystuff.inifile[50]='\0';
    }
    else if(!strcmp((char*)"--perftest", argv[i]))
    {
      read_config(&mystuff);
      init_CL(mystuff.num_streams, devicenumber);
      if ((i+1)<argc)
        tmp = (int)strtol(argv[i+1],&ptr,10);
      else
        tmp = 0;
      perftest(tmp);  
      return 0;
    }
    else if(!strcmp((char*)"--timertest", argv[i]))
    {
      timertest();
      return 0;
    }
    else if(!strcmp((char*)"--sleeptest", argv[i]))
    {
      sleeptest();
      return 0;
    }
    else if(!strcmp((char*)"--CLtest", argv[i]))
    {
      read_config(&mystuff);
      CL_test(devicenumber);
      return 0;
    }
    else
    {
      fprintf(stderr, "ERROR: unknown option '%s'\n", argv[i]);
      return 1;
    }
    i++;
  }

  printf("%s (%dbit build)\n\n", MFAKTO_VERSION, (int)(sizeof(void*)*8));

  read_config(&mystuff);
  
/* print current configuration */
  printf("Compiletime options\n");
#ifdef SIEVE_SIZE_LIMIT
  printf("  SIEVE_SIZE_LIMIT          %dkiB\n", SIEVE_SIZE_LIMIT);
  printf("  SIEVE_SIZE                %dbits\n", SIEVE_SIZE);
#endif
  printf("  SIEVE_SPLIT               %d\n", SIEVE_SPLIT);
  if(SIEVE_SPLIT > SIEVE_PRIMES_MIN)
  {
    printf("ERROR: SIEVE_SPLIT must be <= SIEVE_PRIMES_MIN\n");
    return 1;
  }
#ifdef MORE_CLASSES
  printf("  MORE_CLASSES              enabled\n");
#else
  printf("  MORE_CLASSES              disabled\n");
#endif

#ifdef VERBOSE_TIMING
  printf("  VERBOSE_TIMING            enabled (DEBUG option)\n");
#endif
#ifdef USE_DEVICE_PRINTF
  printf("  USE_DEVICE_PRINTF         enabled (DEBUG option)\n");
#endif
#ifdef CHECKS_MODBASECASE
  printf("  CHECKS_MODBASECASE        enabled (DEBUG option)\n");
#endif
#ifdef DEBUG_STREAM_SCHEDULE
  printf("  DEBUG_STREAM_SCHEDULE     enabled (DEBUG option)\n");
#endif
#ifdef DEBUG_STREAM_SCHEDULE_CHECK
  printf("  DEBUG_STREAM_SCHEDULE_CHECK\n                            enabled (DEBUG option)\n");
#endif
#ifdef RAW_GPU_BENCH
  printf("  RAW_GPU_BENCH             enabled (DEBUG option)\n");
#endif
#ifdef DETAILED_INFO
  printf("  DETAILED_INFO             enabled (DEBUG option)\n");
#endif
#ifdef CL_PERFORMANCE_INFO
  printf("  CL_PERFORMANCE_INFO       enabled (DEBUG option)\n");
#endif

  if(init_CL(mystuff.num_streams, devicenumber)!=CL_SUCCESS)
  {
    printf("init_CL(%d, %d) failed\n", mystuff.num_streams, devicenumber);
    return 2;
  }
  printf("\nOpenCL device info\n");
  printf("  name                      %s (%s)\n", deviceinfo.d_name, deviceinfo.v_name);
  printf("  device (driver) version   %s (%s)\n", deviceinfo.d_ver, deviceinfo.dr_version);
  printf("  maximum threads per block %d\n", (int)deviceinfo.maxThreadsPerBlock);
  printf("  maximum threads per grid  %d\n", (int)deviceinfo.maxThreadsPerGrid);
  printf("  number of multiprocessors %d (%d compute elements (estimate for ATI GPUs))\n", deviceinfo.units, deviceinfo.units * 80);
  printf("  clock rate                %dMHz\n", deviceinfo.max_clock);

  printf("\nAutomatic parameters\n");

  mystuff.threads_per_grid = mystuff.threads_per_grid_max;
  if(mystuff.threads_per_grid > deviceinfo.maxThreadsPerGrid)
  {
    mystuff.threads_per_grid = (cl_uint)deviceinfo.maxThreadsPerGrid;
  }
  printf("  threads per grid          %d\n\n", mystuff.threads_per_grid);

  if (init_CLstreams())
  {
    printf("ERROR: init_CLstreams (malloc buffers?) failed\n");
    return 1;
  }

#ifdef VERBOSE_TIMING
  timer_init(&timer);
#endif
#ifdef SIEVE_SIZE_LIMIT
  sieve_init();
#else
  sieve_init(mystuff.sieve_size, mystuff.sieve_primes_max_global);
#endif
#ifdef VERBOSE_TIMING
  printf("tf(): time spent for sieve_init(): %" PRIu64 "ms\n",timer_diff(&timer)/1000);
#endif

  mystuff.sieve_primes_max = mystuff.sieve_primes_max_global;
  if(mystuff.mode == MODE_NORMAL)
  {

/* before we start real work run a small selftest */  
    mystuff.mode = MODE_SELFTEST_SHORT;
    printf("running a simple selftest ...\n");
    if (selftest(&mystuff, MODE_SELFTEST_SHORT) != 0) return 1; /* selftest failed :( */
    mystuff.mode = MODE_NORMAL;
    /* allow for ^C */
    register_signal_handler(&mystuff);
 
    do
    {
      if (use_worktodo) parse_ret = get_next_assignment(mystuff.workfile, &exp, &bit_min, &bit_max);
      if (parse_ret == 0)
      {
        printf("got assignment: exp=%u bit_min=%d bit_max=%d\n",exp,bit_min,bit_max);
        if (mystuff.p_par[EXP].pos) sprintf(mystuff.p_par[EXP].out, "%d", exp);

        bit_min_stage = bit_min;
        bit_max_stage = bit_max;

        mystuff.sieve_primes_max = sieve_sieve_primes_max(exp, mystuff.sieve_primes_max_global);

        if(mystuff.sieve_primes > mystuff.sieve_primes_max)
        {
          mystuff.sieve_primes = mystuff.sieve_primes_max;
          printf("WARNING: SievePrimes is too big for the current assignment, lowering to %u\n", mystuff.sieve_primes_max);
          printf("         It is not allowed to sieve primes which are equal or bigger than the \n");
          printf("         exponent itself!\n");
        }
        if (mystuff.p_par[SIEVE_PRIMES].pos) sprintf(mystuff.p_par[SIEVE_PRIMES].out, "%7d", mystuff.sieve_primes);

        if(mystuff.stages == 1)
        {
          while( ((calculate_k(exp,bit_max_stage) - calculate_k(exp,bit_min_stage)) > (250000000ULL * NUM_CLASSES))
              && ((bit_max_stage - bit_min_stage) > 1) )  bit_max_stage--;
        }
        tmp = 0;
        while(bit_max_stage <= bit_max && !mystuff.quit)
        {
          if (mystuff.p_par[LOWER_LIMIT].pos) sprintf(mystuff.p_par[LOWER_LIMIT].out, "%2d", bit_min_stage);
          if (mystuff.p_par[UPPER_LIMIT].pos) sprintf(mystuff.p_par[UPPER_LIMIT].out, "%2d", bit_max_stage);
          tmp = tf(exp, bit_min_stage, bit_max_stage, &mystuff, 0, 0, AUTOSELECT_KERNEL);
          if(tmp == RET_ERROR) return 1; /* bail out, we might have a serios problem  */

          if(tmp != RET_QUIT)
          {

            if( (mystuff.stopafterfactor > 0) && (tmp > 0) )
            {
              bit_max_stage = bit_max;
            }

            if(use_worktodo)
            {
              if(bit_max_stage == bit_max)parse_ret = clear_assignment(mystuff.workfile, exp, bit_min_stage, bit_max, 0);
              else                        parse_ret = clear_assignment(mystuff.workfile, exp, bit_min_stage, bit_max, bit_max_stage);

                   if(parse_ret == 3) printf("ERROR: clear_assignment() / modify_assignment(): can't open \"%s\"\n", mystuff.workfile);
              else if(parse_ret == 4) printf("ERROR: clear_assignment() / modify_assignment(): can't open \"__worktodo__.tmp\"\n");
              else if(parse_ret == 5) printf("ERROR: clear_assignment() / modify_assignment(): assignment not found in \"%s\"\n", mystuff.workfile);
              else if(parse_ret == 6) printf("ERROR: clear_assignment() / modify_assignment(): can't rename workfiles\n");
              else if(parse_ret != 0) printf("ERROR: clear_assignment() / modify_assignment(): Unknown error (%d)\n", parse_ret);
            }

            bit_min_stage = bit_max_stage;
            bit_max_stage++;
          }
        }
      }
      else if(parse_ret == 1) printf("ERROR: get_next_assignment(): can't open \"%s\"\n", mystuff.workfile);
      else if(parse_ret == 2) printf("ERROR: get_next_assignment(): no valid assignment found in \"%s\"\n", mystuff.workfile);
      else if(parse_ret != 0) printf("ERROR: get_next_assignment(): Unknown error (%d)\n", parse_ret);
    }
    while(parse_ret == 0 && use_worktodo && !mystuff.quit);
  }
  else // mystuff.mode != MODE_NORMAL
  {
    if (0 != selftest(&mystuff, mystuff.mode))
    {
      printf ("Error exit as selftest failed\n");
      cleanup_CL();
      sieve_free();
      return 1;
    }
  }

  cleanup_CL();

  sieve_free();

  return 0;
}
