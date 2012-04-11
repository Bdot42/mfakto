/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2011  Oliver Weihe (o.weihe@t-online.de)
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
#include <string.h>
#if defined(BUILD_OPENCL)
#include <CL/cl.h>
#endif

#include "params.h"
#include "my_types.h"

extern kernel_info_t       kernel_info[];

int my_read_int(char *inifile, char *name, int *value)
{
  FILE *in;
  char buf[100];
  int found=0;

  in=fopen(inifile,"r");
  if(!in)return 1;
  while(fgets(buf,100,in) && !found)
  {
    if(!strncmp(buf,name,strlen(name)) && buf[strlen(name)]=='=')
    {
      if(sscanf(&(buf[strlen(name)+1]),"%d",value)==1)found=1;
    }
  }
  fclose(in);
  if(found)return 0;
  return 1;
}

int my_read_string(char *inifile, char *name, char *string)
{
  FILE *in;
  char buf[100];
  int found=0;

  in=fopen(inifile,"r");
  if(!in)return 1;
  while(fgets(buf,100,in) && !found)
  {
    if(!strncmp(buf,name,strlen(name)) && buf[strlen(name)]=='=')
    {
      if(sscanf(&(buf[strlen(name)+1]),"%50s",string)==1)found=1;	/* string has enough space for 50+1 chars, see my_types.h */
    }
  }
  fclose(in);
  if(found)return 0;
  return 1;
}

int read_config(mystuff_t *mystuff)
{
  int i;
  char tmp[51];

  printf("\nRuntime options\n");
  printf("  Inifile                   %s\n",mystuff->inifile);

/*****************************************************************************/  

  if(my_read_int(mystuff->inifile, "SievePrimesMax", &i))
  {
    printf("WARNING: Cannot read SievePrimesMax from inifile, using default value (%d)\n", SIEVE_PRIMES_MAX);
    i=SIEVE_PRIMES_MAX;
  }
  else if((i < 5000) || (i > 1000000))
  {
    printf("WARNING: SievePrimesMax must be between 5000 and 1000000, using default value (%d)\n", SIEVE_PRIMES_MAX);
    i=SIEVE_PRIMES_MAX;
  }
  printf("  SievePrimesMax            %d\n",i);
  mystuff->sieve_primes_max_global = i;

/*****************************************************************************/
  if(my_read_int(mystuff->inifile, "SievePrimes", &i))
  {
    printf("WARNING: Cannot read SievePrimes from inifile, using default value (%d)\n",SIEVE_PRIMES_DEFAULT);
    i=SIEVE_PRIMES_DEFAULT;
  }
  else
  {
    if((cl_uint)i>mystuff->sieve_primes_max_global)
    {
      printf("WARNING: Read SievePrimes=%d from inifile, using max value (%d)\n",i,mystuff->sieve_primes_max_global);
      i=mystuff->sieve_primes_max_global;
    }
    else if(i<SIEVE_PRIMES_MIN)
    {
      printf("WARNING: Read SievePrimes=%d from inifile, using min value (%d)\n",i,SIEVE_PRIMES_MIN);
      i=SIEVE_PRIMES_MIN;
    }
  }
  printf("  SievePrimes               %d\n",i);
  mystuff->sieve_primes = i;

/*****************************************************************************/  

  if(my_read_int(mystuff->inifile, "SievePrimesAdjust", &i))
  {
    printf("WARNING: Cannot read SievePrimesAdjust from inifile, using default value (0)\n");
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: SievePrimesAdjust must be 0 or 1, using default value (0)\n");
    i=0;
  }
  printf("  SievePrimesAdjust         %d\n",i);
  mystuff->sieve_primes_adjust = i;
  if (mystuff->sieve_primes_adjust == 0)
    mystuff->sieve_primes_max_global = mystuff->sieve_primes;  // no chance to use higher primes

/*****************************************************************************/  
#ifdef SIEVE_SIZE_LIMIT
  mystuff->sieve_size = SIEVE_SIZE;
#else
  if(my_read_int(mystuff->inifile, "SieveSizeLimit", &i))
  {
    printf("WARNING: Cannot read SieveSizeLimit from inifile, using default value (32)\n");
    i=32;
  }
  else if(i <= 13*17*19*23/8192)
  {
    printf("WARNING: SieveSizeLimit must be > %d, using default value (32)\n", 13*17*19*23/8192);
    i=32;
  }
  printf("  SieveSizeLimit            %d kiB\n", i);
  mystuff->sieve_size = ((i<<13) - (i<<13) % (13*17*19*23));
  printf("  SieveSize                 %d bits\n", mystuff->sieve_size);
#endif
/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "NumStreams", &i))
  {
    printf("WARNING: Cannot read NumStreams from inifile, using default value (%d)\n",NUM_STREAMS_DEFAULT);
    i=NUM_STREAMS_DEFAULT;
  }
  else
  {
    if(i>NUM_STREAMS_MAX)
    {
      printf("WARNING: Read NumStreams=%d from inifile, using max value (%d)\n",i,NUM_STREAMS_MAX);
      i=NUM_STREAMS_MAX;
    }
    else if(i<NUM_STREAMS_MIN)
    {
      printf("WARNING: Read NumStreams=%d from inifile, using min value (%d)\n",i,NUM_STREAMS_MIN);
      i=NUM_STREAMS_MIN;
    }
  }
  printf("  NumStreams                %d\n",i);
  mystuff->num_streams = i;

/*****************************************************************************/

/* CPU streams not used by mfakto
  if(my_read_int(mystuff->inifile, "CPUStreams", &i))
  {
    printf("WARNING: Cannot read CPUStreams from inifile, using default value (%d)\n",CPU_STREAMS_DEFAULT);
    i=CPU_STREAMS_DEFAULT;
  }
  else
  {
    if(i>CPU_STREAMS_MAX)
    {
      printf("WARNING: Read CPUStreams=%d from inifile, using max value (%d)\n",i,CPU_STREAMS_MAX);
      i=CPU_STREAMS_MAX;
    }
    else if(i<CPU_STREAMS_MIN)
    {
      printf("WARNING: Read CPUStreams=%d from inifile, using min value (%d)\n",i,CPU_STREAMS_MIN);
      i=CPU_STREAMS_MIN;
    }
  }
  printf("  CPUStreams                %d\n",i);
  mystuff->cpu_streams = i;
  */
/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "GridSize", &i))
  {
    printf("WARNING: Cannot read GridSize from inifile, using default value (3)\n");
    i = 3;
  }
  else
  {
    if(i > 4)
    {
      printf("WARNING: Read GridSize=%d from inifile, using max value (4)\n", i);
      i = 4;
    }
    else if(i < 0)
    {
      printf("WARNING: Read GridSize=%d from inifile, using min value (0)\n", i);
      i = 0;
    }
  }
  printf("  GridSize                  %d\n",i);
       if(i == 0)  mystuff->threads_per_grid_max =  131072;
  else if(i == 1)  mystuff->threads_per_grid_max =  262144;
  else if(i == 2)  mystuff->threads_per_grid_max =  524288;
  else if(i == 3)  mystuff->threads_per_grid_max = 1048576;
  else             mystuff->threads_per_grid_max = 2097152;

/*****************************************************************************/

  if(my_read_string(mystuff->inifile, "WorkFile", mystuff->workfile))
  {
    sprintf(mystuff->workfile, "worktodo.txt");
    printf("WARNING: Cannot read WorkFile from inifile, using default (%s)\n", mystuff->workfile);
  }
  printf("  WorkFile                  %s\n", mystuff->workfile);

/*****************************************************************************/

  if(my_read_string(mystuff->inifile, "ResultsFile", mystuff->resultsfile))
  {
    printf("WARNING: Cannot read ResultsFile from inifile, using default (results.txt)\n");
    sprintf(mystuff->resultsfile, "results.txt");
  }
  printf("  ResultsFile               %s\n", mystuff->resultsfile);

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "Checkpoints", &i))
  {
    printf("WARNING: Cannot read Checkpoints from inifile, enabled by default\n");
    i=1;
  }
  else if(i < 0)
  {
    printf("WARNING: Checkpoints must be 0 (disabled) or greater, enabled by default\n");
    i=1;
  }
  if(i==0)printf("  Checkpoints               disabled\n");
  else if (i==1) printf("  Checkpoints               enabled\n");
  else           printf("  Checkpoints               every %d classes\n", i);
  mystuff->checkpoints = i;

/*****************************************************************************/
  if (mystuff->checkpoints > 1)
  {
    printf("  CheckpointDelay           ignored as Checkpoints > 1\n");
    mystuff->checkpointdelay = 300;
  }
  else
  {
    if(my_read_int(mystuff->inifile, "CheckpointDelay", &i))
    {
      printf("WARNING: Cannot read CheckpointDelay from inifile, set to 300s by default\n");
      i = 300;
    }
    if(i > 3600)
    {
      printf("WARNING: Maximum value for CheckpointDelay is 3600s\n");
      i = 3600;
    }
    if(i < 0)
    {
      printf("WARNING: Minimum value for CheckpointDelay is 0s\n");
      i = 0;
    }
    printf("  CheckpointDelay           %ds\n", i);
    mystuff->checkpointdelay = i;
  }

/*****************************************************************************/
  
  if(my_read_int(mystuff->inifile, "Stages", &i))
  {
    printf("WARNING: Cannot read Stages from inifile, enabled by default\n");
    i=1;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: Stages must be 0 or 1, enabled by default\n");
    i=1;
  }
  if(i==0)printf("  Stages                    disabled\n");
  else    printf("  Stages                    enabled\n");
  mystuff->stages = i;

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "StopAfterFactor", &i))
  {
    printf("WARNING: Cannot read StopAfterFactor from inifile, set to 1 by default\n");
    i=1;
  }
  else if( (i < 0) || (i > 2) )
  {
    printf("WARNING: StopAfterFactor must be 0, 1 or 2, set to 1 by default\n");
    i=1;
  }
       if(i==0)printf("  StopAfterFactor           disabled\n");
  else if(i==1)printf("  StopAfterFactor           bitlevel\n");
  else if(i==2)printf("  StopAfterFactor           class\n");
  mystuff->stopafterfactor = i;

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "PrintMode", &i))
  {
    printf("WARNING: Cannot read PrintMode from inifile, set to 0 by default\n");
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: PrintMode must be 0 or 1, set to 0 by default\n");
    i=0;
  }
  if(i == 0)printf("  PrintMode                 full\n");
  else      printf("  PrintMode                 compact\n");
  mystuff->printmode = i;

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "AllowSleep", &i))
  {
    printf("WARNING: Cannot read AllowSleep from inifile, set to 0 by default\n");
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: AllowSleep must be 0 or 1, set to 0 by default\n");
    i=0;
  }
  if(i == 0)printf("  AllowSleep                no\n");
  else      printf("  AllowSleep                yes\n");
  mystuff->allowsleep = i;

  /*****************************************************************************/

  if(my_read_int(mystuff->inifile, "VectorSize", &i))
  {
    printf("WARNING: Cannot read VectorSize from inifile, set to 4 by default\n");
    i=4;
  }
  else if(i != 1 && i != 2 && i != 4 && i != 8 && i != 16 )
  {
    printf("WARNING: VectorSize must be one of 1, 2, 4, or 8, set to 4 by default\n");
    i=4;
  }
#ifdef CHECKS_MODBASECASE
  if (i>1)
  {
    printf("WARNING: Reducing vector size from %d to 1 due to CHECKS_MODBASECASE.\n", i);
    i=1;
  }
#endif
  printf("  VectorSize                %d\n", i);
  mystuff->vectorsize = i;

/*****************************************************************************/

  mystuff->preferredKernel = BARRETT79_MUL32;

  if (my_read_string(mystuff->inifile, "PreferKernel", tmp))
  {
    printf("WARNING: Cannot read PreferKernel from inifile, using default (mfakto_cl_barrett79)\n");
  }
  else if (strcmp(tmp, "mfakto_cl_71") == 0)
  {
    mystuff->preferredKernel = _71BIT_MUL24;
  }
  else if (strcmp(tmp, "mfakto_cl_barrett79") != 0)
  {
    printf("WARNING: Unknown setting \"%s\" for PreferKernel, using default (mfakto_cl_barrett79)\n", tmp);
  }

  printf("  PreferKernel              %s\n", kernel_info[mystuff->preferredKernel].kernelname);

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "SieveOnGPU", &i))
  {
    printf("WARNING: Cannot read SieveOnGPU from inifile, set to 0 by default\n");
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: SieveOnGPU must be 0 or 1, set to 0 by default\n");
    i=0;
  }
  if(i == 0)printf("  SieveOnGPU                no\n");
  else      printf("  SieveOnGPU                yes\n");
  mystuff->sieve_gpu = i;

  /*****************************************************************************/

  if(my_read_int(mystuff->inifile, "SmallExp", &i))
  {
    printf("WARNING: Cannot read SmallExp from inifile, set to 0 by default\n");
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: SmallExp must be 0 or 1, set to 0 by default\n");
    i=0;
  }
  if(i == 0)printf("  SmallExp                  no\n");
  else      printf("  SmallExp                  yes\n");
  mystuff->small_exp = i;

  /*****************************************************************************/

  return 0;
}
