/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)

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

#include <stdio.h>
#include <string.h>
#if defined(BUILD_OPENCL)
#include <CL/cl.h>
#endif

#include "params.h"
#include "my_types.h"

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
  printf("\nRuntime options\n");

  if(my_read_int("mfaktc.ini", "SievePrimes", &i))
  {
    printf("WARNING: Cannot read SievePrimes from mfaktc.ini, using default value (%d)\n",SIEVE_PRIMES_DEFAULT);
    i=SIEVE_PRIMES_DEFAULT;
  }
  else
  {
    if(i>SIEVE_PRIMES_MAX)
    {
      printf("WARNING: Read SievePrimes=%d from mfaktc.ini, using max value (%d)\n",i,SIEVE_PRIMES_MAX);
      i=SIEVE_PRIMES_MAX;
    }
    else if(i<SIEVE_PRIMES_MIN)
    {
      printf("WARNING: Read SievePrimes=%d from mfaktc.ini, using min value (%d)\n",i,SIEVE_PRIMES_MIN);
      i=SIEVE_PRIMES_MIN;
    }
  }
  printf("  SievePrimes               %d\n",i);
  mystuff->sieve_primes = i;

/*****************************************************************************/  

  if(my_read_int("mfaktc.ini", "SievePrimesAdjust", &i))
  {
    printf("WARNING: Cannot read SievePrimesAdjust from mfaktc.ini, using default value (0)\n");
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: SievePrimesAdjust must be 0 or 1, using default value (0)\n");
    i=0;
  }
  printf("  SievePrimesAdjust         %d\n",i);
  mystuff->sieve_primes_adjust = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "NumStreams", &i))
  {
    printf("WARNING: Cannot read NumStreams from mfaktc.ini, using default value (%d)\n",NUM_STREAMS_DEFAULT);
    i=NUM_STREAMS_DEFAULT;
  }
  else
  {
    if(i>NUM_STREAMS_MAX)
    {
      printf("WARNING: Read NumStreams=%d from mfaktc.ini, using max value (%d)\n",i,NUM_STREAMS_MAX);
      i=NUM_STREAMS_MAX;
    }
    else if(i<NUM_STREAMS_MIN)
    {
      printf("WARNING: Read NumStreams=%d from mfaktc.ini, using min value (%d)\n",i,NUM_STREAMS_MIN);
      i=NUM_STREAMS_MIN;
    }
  }
  printf("  NumStreams                %d\n",i);
  mystuff->num_streams = i;

/*****************************************************************************/

/* CPU streams not used by mfakto
  if(my_read_int("mfaktc.ini", "CPUStreams", &i))
  {
    printf("WARNING: Cannot read CPUStreams from mfaktc.ini, using default value (%d)\n",CPU_STREAMS_DEFAULT);
    i=CPU_STREAMS_DEFAULT;
  }
  else
  {
    if(i>CPU_STREAMS_MAX)
    {
      printf("WARNING: Read CPUStreams=%d from mfaktc.ini, using max value (%d)\n",i,CPU_STREAMS_MAX);
      i=CPU_STREAMS_MAX;
    }
    else if(i<CPU_STREAMS_MIN)
    {
      printf("WARNING: Read CPUStreams=%d from mfaktc.ini, using min value (%d)\n",i,CPU_STREAMS_MIN);
      i=CPU_STREAMS_MIN;
    }
  }
  printf("  CPUStreams                %d\n",i);
  mystuff->cpu_streams = i;
  */
/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "GridSize", &i))
  {
    printf("WARNING: Cannot read GridSize from mfaktc.ini, using default value (3)\n");
    i = 3;
  }
  else
  {
    if(i > 4)
    {
      printf("WARNING: Read GridSize=%d from mfaktc.ini, using max value (4)\n", i);
      i = 4;
    }
    else if(i < 0)
    {
      printf("WARNING: Read GridSize=%d from mfaktc.ini, using min value (0)\n", i);
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

  if(my_read_string("mfaktc.ini", "WorkFile", mystuff->workfile))
  {
    printf("WARNING: can't read WorkFile from mfaktc.ini, using default (worktodo.ini)\n");
    sprintf(mystuff->workfile, "worktodo.ini");
  }
  printf("  WorkFile                  %s\n", mystuff->workfile);

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "Checkpoints", &i))
  {
    printf("WARNING: Cannot read Checkpoints from mfaktc.ini, enabled by default\n");
    i=1;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: Checkpoints must be 0 or 1, enabled by default\n");
    i=1;
  }
  if(i==0)printf("  Checkpoints               disabled\n");
  else    printf("  Checkpoints               enabled\n");
  mystuff->checkpoints = i;

/*****************************************************************************/

  if(my_read_int("mfaktc.ini", "Stages", &i))
  {
    printf("WARNING: Cannot read Stages from mfaktc.ini, enabled by default\n");
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

  if(my_read_int("mfaktc.ini", "StopAfterFactor", &i))
  {
    printf("WARNING: Cannot read StopAfterFactor from mfaktc.ini, set to 1 by default\n");
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

  if(my_read_int("mfaktc.ini", "PrintMode", &i))
  {
    printf("WARNING: Cannot read PrintMode from mfaktc.ini, set to 0 by default\n");
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

  if(my_read_int("mfaktc.ini", "AllowSleep", &i))
  {
    printf("WARNING: Cannot read AllowSleep from mfaktc.ini, set to 0 by default\n");
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

  return 0;
}
