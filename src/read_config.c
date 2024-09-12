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
*/

#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#define strcasecmp _stricmp
#endif
#include <errno.h>
#if defined BUILD_OPENCL
  #if defined __APPLE__
    #include <OpenCL/cl.h>
  #else
    #include <CL/cl.h>
  #endif
#endif

#include "params.h"
#include "my_types.h"
#include "output.h"

extern kernel_info_t   kernel_info[];
extern GPU_type        gpu_types[];
static int inifile_unavailable = 0;

int my_read_int(char *inifile, char *name, int *value)
{
  FILE *in;
  char buf[100];
  int found=0;

  in=fopen(inifile,"r");
  if(!in)
  {
    if (!inifile_unavailable)
    {
      char msg[80];
      inifile_unavailable = 1;
      sprintf(msg, "Cannot load INI file \"%.55s\"", inifile);
      perror(msg);
    }
    return 1;
  }
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

static int my_read_ulong(char *inifile, char *name, unsigned long long int *value)
{
  FILE *in;
  char buf[100];
  int found=0;

  in=fopen(inifile,"r");
  if(!in)
  {
    if (!inifile_unavailable)
    {
      char msg[80];
      inifile_unavailable = 1;
      sprintf(msg, "Cannot load INI file \"%.55s\"", inifile);
      perror(msg);
    }
    return 1;
  }
  while(fgets(buf,100,in) && !found)
  {
    if(!strncmp(buf,name,strlen(name)) && buf[strlen(name)]=='=')
    {
      #ifdef __MINGW32__
        if(sscanf(&(buf[strlen(name)+1]),"%I64u",value)==1)found=1;
      #else
        if(sscanf(&(buf[strlen(name)+1]),"%llu",value)==1)found=1;
      #endif
    }
  }
  fclose(in);
  if(found)return 0;
  return 1;
}

int my_read_string(char *inifile, char *name, char *string, unsigned int len)
{
  FILE *in;
  char buf[512];
  unsigned int found=0;
  unsigned int idx = (unsigned int) strlen(name);

  in=fopen(inifile,"r");
  if(!in)
  {
    if (!inifile_unavailable)
    {
      char msg[80];
      inifile_unavailable = 1;
      sprintf(msg, "Cannot load INI file \"%.55s\"", inifile);
      perror(msg);
    }
    return 1;
  }
  while(fgets(buf,512,in) && !found)
  {
    if(!strncmp(buf,name,idx) && buf[idx]=='=')
    {
      found = (unsigned int) strlen(buf + idx + 1);
      found = (len > found ? found : len) - 1;
      if (found)
      {
        strncpy(string, buf + idx + 1, found);
        if(string[found - 1] == '\r') found--; //remove '\r' from string, this happens when reading a DOS/Windows formatted file on Linux
      }
      string[found]='\0';
    }
  }
  fclose(in);
  if(found>0)return 0;
  return 1;
}


int read_config(mystuff_t *mystuff)
{
  int i;
  char tmp[51];
  unsigned long long int ul;

  if(mystuff->verbosity == -1 || mystuff->override_v == 0)
  {
    if(my_read_int(mystuff->inifile, "Verbosity", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read Verbosity from INI file, set to 1 by default\n");
      mystuff->verbosity = 1;
    }
    else
      mystuff->verbosity = i;
  }

  if(mystuff->verbosity >= 1)logprintf(mystuff, 
                                    "\nRuntime options\n"
                                    "  INI file                  %s\n"
                                    "  Verbosity                 %d\n", mystuff->inifile, mystuff->verbosity);

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "SieveOnGPU", &i))
  {
    logprintf(mystuff, "WARNING: Cannot read SieveOnGPU from INI file, set to 0 by default\n");
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    logprintf(mystuff, "WARNING: SieveOnGPU must be 0 or 1, set to 0 by default\n");
    i=0;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i == 0)logprintf(mystuff, "  SieveOnGPU                no\n");
    else      logprintf(mystuff, "  SieveOnGPU                yes\n");
  }
  mystuff->gpu_sieving = i;

/*****************************************************************************/

  if (mystuff->gpu_sieving == 0)
  {
    if(mystuff->verbosity >= 1) logprintf(mystuff, "  MoreClasses               yes (due to CPU-sieving)\n");
    mystuff->more_classes = 1;
    mystuff->num_classes  = 4620;

    if(my_read_int(mystuff->inifile, "SievePrimesMin", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read SievePrimesMin from INI file, using default value (%d)\n", 5000);
      i = 5000;
    }
    else if((i < SIEVE_PRIMES_MIN) || (i >= SIEVE_PRIMES_MAX))
    {
      logprintf(mystuff, "WARNING: SievePrimesMin must be between %d and %d, using default value (%d)\n",
          SIEVE_PRIMES_MIN, SIEVE_PRIMES_MAX, 5000);
      i = 5000;
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  SievePrimesMin            %d\n",i);
    mystuff->sieve_primes_min = i;

  /*****************************************************************************/

    if(my_read_int(mystuff->inifile, "SievePrimesMax", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read SievePrimesMax from INI file, using default value (%d)\n", 200000);
      i = 200000;
    }
    else if((i < (int) mystuff->sieve_primes_min) || (i > SIEVE_PRIMES_MAX))
    {
      logprintf(mystuff, "WARNING: SievePrimesMax must be between SievePrimesMin(%d) and %d, using default value (%d)\n",
          mystuff->sieve_primes_min, SIEVE_PRIMES_MAX, 200000);
      i = 200000;
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  SievePrimesMax            %d\n",i);
    mystuff->sieve_primes_max = i;

  /*****************************************************************************/
    if(my_read_int(mystuff->inifile, "SievePrimes", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read SievePrimes from INI file, using default value (%d)\n", SIEVE_PRIMES_DEFAULT);
      i = SIEVE_PRIMES_DEFAULT;
    }
    else
    {
      if((cl_uint)i>mystuff->sieve_primes_max)
      {
        logprintf(mystuff, "WARNING: Read SievePrimes=%d from INI file, using max value (%d)\n", i, mystuff->sieve_primes_max);
        i = mystuff->sieve_primes_max;
      }
      else if( i < (int) mystuff->sieve_primes_min)
      {
        logprintf(mystuff, "WARNING: Read SievePrimes=%d from INI file, using min value (%d)\n", i, mystuff->sieve_primes_min);
        i = mystuff->sieve_primes_min;
      }
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  SievePrimes               %d\n",i);
    mystuff->sieve_primes = i;

  /*****************************************************************************/

    if(my_read_int(mystuff->inifile, "SievePrimesAdjust", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read SievePrimesAdjust from INI file, using default value (0)\n");
      i = 0;
    }
    else if(i != 0 && i != 1)
    {
      logprintf(mystuff, "WARNING: SievePrimesAdjust must be 0 or 1, using default value (0)\n");
      i = 0;
    }
    if(mystuff->verbosity >= 1) logprintf(mystuff, "  SievePrimesAdjust         %d\n",i);
    mystuff->sieve_primes_adjust = i;
    if (mystuff->sieve_primes_adjust == 0)
      mystuff->sieve_primes_max = mystuff->sieve_primes;  // no chance to use higher primes

  /*****************************************************************************/
#ifdef SIEVE_SIZE_LIMIT
    mystuff->sieve_size = SIEVE_SIZE;
#else
    if(my_read_int(mystuff->inifile, "SieveSizeLimit", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read SieveSizeLimit from INI file, using default value (32)\n");
      i=32;
    }
    else if(i <= 13*17*19*23/8192)
    {
      logprintf(mystuff, "WARNING: SieveSizeLimit must be > %d, using default value (32)\n", 13*17*19*23/8192);
      i=32;
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  SieveSizeLimit            %d kiB\n", i);
    mystuff->sieve_size = ((i<<13) - (i<<13) % (13*17*19*23));
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  SieveSize                 %d bits\n", mystuff->sieve_size);
#endif

  /*****************************************************************************/

    if(my_read_int(mystuff->inifile, "NumStreams", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read NumStreams from INI file, using default value (%d)\n",NUM_STREAMS_DEFAULT);
      i=NUM_STREAMS_DEFAULT;
    }
    else
    {
      if(i>NUM_STREAMS_MAX)
      {
        logprintf(mystuff, "WARNING: Read NumStreams=%d from INI file, using max value (%d)\n",i,NUM_STREAMS_MAX);
        i=NUM_STREAMS_MAX;
      }
      else if(i<NUM_STREAMS_MIN)
      {
        logprintf(mystuff, "WARNING: Read NumStreams=%d from INI file, using min value (%d)\n",i,NUM_STREAMS_MIN);
        i=NUM_STREAMS_MIN;
      }
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  NumStreams                %d\n",i);
    mystuff->num_streams = i;

  /*****************************************************************************/

  /* CPU streams not used by mfakto
    if(my_read_int(mystuff->inifile, "CPUStreams", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read CPUStreams from inifile, using default value (%d)\n",CPU_STREAMS_DEFAULT);
      i=CPU_STREAMS_DEFAULT;
    }
    else
    {
      if(i>CPU_STREAMS_MAX)
      {
        logprintf(mystuff, "WARNING: Read CPUStreams=%d from inifile, using max value (%d)\n",i,CPU_STREAMS_MAX);
        i=CPU_STREAMS_MAX;
      }
      else if(i<CPU_STREAMS_MIN)
      {
        logprintf(mystuff, "WARNING: Read CPUStreams=%d from inifile, using min value (%d)\n",i,CPU_STREAMS_MIN);
        i=CPU_STREAMS_MIN;
      }
    }
    logprintf(mystuff, "  CPUStreams                %d\n",i);
    mystuff->cpu_streams = i;
    */
  /*****************************************************************************/

    if(my_read_int(mystuff->inifile, "GridSize", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read GridSize from INI file, using default value (3)\n");
      i = 3;
    }
    else
    {
      if(i > 4)
      {
        logprintf(mystuff, "WARNING: Read GridSize=%d from INI file, using max value (4)\n", i);
        i = 4;
      }
      else if(i < 0)
      {
        logprintf(mystuff, "WARNING: Read GridSize=%d from INI file, using min value (0)\n", i);
        i = 0;
      }
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  GridSize                  %d\n",i);
         if(i == 0)  mystuff->threads_per_grid_max =  131072;
    else if(i == 1)  mystuff->threads_per_grid_max =  262144;
    else if(i == 2)  mystuff->threads_per_grid_max =  524288;
    else if(i == 3)  mystuff->threads_per_grid_max = 1048576;
    else             mystuff->threads_per_grid_max = 2097152;

  /*****************************************************************************/

    if(my_read_ulong(mystuff->inifile, "SieveCPUMask", &ul))
    {
      logprintf(mystuff, "WARNING: Cannot read SieveCPUMask from INI file, set to 0 by default\n");
      ul=0;
    }
    #ifdef __MINGW32__
      if(mystuff->verbosity >= 1)logprintf(mystuff, "  SieveCPUMask              %I64u\n", ul);
    #else
      if(mystuff->verbosity >= 1)logprintf(mystuff, "  SieveCPUMask              %lld\n", ul);
    #endif

    mystuff->cpu_mask = ul;
  /*****************************************************************************/
  /* not used in mfakto (yet)
    if(my_read_int(mystuff->inifile, "AllowSleep", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read AllowSleep from inifile, set to 0 by default\n");
      i=0;
    }
    else if(i != 0 && i != 1)
    {
      logprintf(mystuff, "WARNING: AllowSleep must be 0 or 1, set to 0 by default\n");
      i=0;
    }
    if(mystuff->verbosity >= 1)
    {
      if(i == 0)logprintf(mystuff, "  AllowSleep                no\n");
      else      logprintf(mystuff, "  AllowSleep                yes\n");
    }
    mystuff->allowsleep = i;
    */
/*****************************************************************************/

  }
  else // SieveOnGPU
  {
    mystuff->num_streams = 3; // GPU sieve always uses only one stream, but perftest may use more
    mystuff->threads_per_grid_max = 2097152; // not used for the GPU sieve - defined here to satisfy some calculations

    if(my_read_int(mystuff->inifile, "MoreClasses", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read MoreClasses from INI file, set to 1 by default\n");
      i=1;
    }
    else if(i != 0 && i != 1)
    {
      logprintf(mystuff, "WARNING: MoreClasses must be 0 or 1, set to 1 by default\n");
      i=1;
    }
    if(mystuff->verbosity >= 1)
    {
      if(i == 0)logprintf(mystuff, "  MoreClasses               no\n");
      else      logprintf(mystuff, "  MoreClasses               yes\n");
    }
    mystuff->more_classes = i;
    mystuff->num_classes  = i?4620:420;

/*****************************************************************************/

    if(my_read_int(mystuff->inifile, "GPUSievePrimes", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read GPUSievePrimes from INI file, using default value (%d)\n",GPU_SIEVE_PRIMES_DEFAULT);
      i = GPU_SIEVE_PRIMES_DEFAULT;
    }
    else
    {
      if(i > GPU_SIEVE_PRIMES_MAX)
      {
        logprintf(mystuff, "WARNING: Read GPUSievePrimes=%d from INI file, using max value (%d)\n",i,GPU_SIEVE_PRIMES_MAX);
        i = GPU_SIEVE_PRIMES_MAX;
      }
      else if(i < GPU_SIEVE_PRIMES_MIN)
      {
        logprintf(mystuff, "WARNING: Read GPUSievePrimes=%d from INI file, using min value (%d)\n",i,GPU_SIEVE_PRIMES_MIN);
        i = GPU_SIEVE_PRIMES_MIN;
      }
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  GPUSievePrimes            %d\n",i);
    mystuff->sieve_primes = i;

/*****************************************************************************/

    if(my_read_int(mystuff->inifile, "GPUSieveProcessSize", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read GPUSieveProcessSize from INI file, using default value (%d)\n",GPU_SIEVE_PROCESS_SIZE_DEFAULT);
      i = GPU_SIEVE_PROCESS_SIZE_DEFAULT;
    }
    else
    {
      if(i % 8 != 0)
      {
        logprintf(mystuff, "WARNING: GPUSieveProcessSize must be a multiple of 8\n");
        i &= 0xFFFFFFF8;
        logprintf(mystuff, "         --> changed GPUSieveProcessSize to %d\n", i);
      }
      if(i > GPU_SIEVE_PROCESS_SIZE_MAX)
      {
        logprintf(mystuff, "WARNING: Read GPUSieveProcessSize=%d from INI file, using max value (%d)\n",i,GPU_SIEVE_PROCESS_SIZE_MAX);
        i = GPU_SIEVE_PROCESS_SIZE_MAX;
      }
      else if(i < GPU_SIEVE_PROCESS_SIZE_MIN)
      {
        logprintf(mystuff, "WARNING: Read GPUSieveProcessSize=%d from INI file, using min value (%d)\n",i,GPU_SIEVE_PROCESS_SIZE_MIN);
        i = GPU_SIEVE_PROCESS_SIZE_MIN;
      }
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  GPUSieveProcessSize       %d Kib\n",i);
    mystuff->gpu_sieve_processing_size = i * 1024;

/*****************************************************************************/

    if(my_read_int(mystuff->inifile, "GPUSieveSize", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read GPUSieveSize from INI file, using default value (%d)\n",GPU_SIEVE_SIZE_DEFAULT);
      i = GPU_SIEVE_SIZE_DEFAULT;
    }
    else
    {
      if(i > GPU_SIEVE_SIZE_MAX)
      {
        logprintf(mystuff, "WARNING: Read GPUSieveSize=%d from INI file, using max value (%d)\n",i,GPU_SIEVE_SIZE_MAX);
        i = GPU_SIEVE_SIZE_MAX;
      }
      else if(i < GPU_SIEVE_SIZE_MIN)
      {
        logprintf(mystuff, "WARNING: Read GPUSieveSize=%d from INI file, using min value (%d)\n",i,GPU_SIEVE_SIZE_MIN);
        i = GPU_SIEVE_SIZE_MIN;
      }
      if (i * 1024 * 1024 % mystuff->gpu_sieve_processing_size != 0)
      {
        // can only happen when GPUSieveProcessSize=24 ==> make i divisible by 3
        logprintf(mystuff, "WARNING: GPUSieveSize=%dM must be a multiple of GPUSieveProcessSize=%dk, ", i, mystuff->gpu_sieve_processing_size / 1024);
        i -= i%3;
        while (i < GPU_SIEVE_SIZE_MIN) i+=3;  // make sure it's not too low
        logprintf(mystuff, "adjusting GPUSieveSize to %dM\n", i);
      }
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  GPUSieveSize              %d Mib\n",i);
    mystuff->gpu_sieve_size = i * 1024 * 1024;

    /*****************************************************************************/

    if(my_read_int(mystuff->inifile, "FlushInterval", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read FlushInterval from INI file, using default value 0\n");
      i = 0;
    }
    else
    {
      if(i < 0)
      {
        logprintf(mystuff, "WARNING: Read FlushInterval=%d from INI file, using min value (0)\n",i);
        i = 0;
      }
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  FlushInterval             %d\n",i);
    mystuff->flush = i;
  } // end GPU sieve only

/*****************************************************************************/

  if(my_read_string(mystuff->inifile, "WorkFile", mystuff->workfile, 50))
  {
    sprintf(mystuff->workfile, "worktodo.txt");
    logprintf(mystuff, "WARNING: Cannot read WorkFile from INI file, using default (%s)\n", mystuff->workfile);
  }
  if(mystuff->verbosity >= 1)logprintf(mystuff, "  WorkFile                  %s\n", mystuff->workfile);

/*****************************************************************************/

  if(my_read_string(mystuff->inifile, "ResultsFile", mystuff->resultfile, 50))
  {
    sprintf(mystuff->resultfile, "results.txt");
    logprintf(mystuff, "WARNING: Cannot read ResultsFile from INI file, using default (%s)\n", mystuff->resultfile);
  }
  if(mystuff->verbosity >= 1)logprintf(mystuff, "  ResultsFile               %s\n", mystuff->resultfile);

 /*****************************************************************************/

  if (my_read_string(mystuff->inifile, "JSONResultsFile", mystuff->jsonresultfile, 50))
  {
    sprintf(mystuff->jsonresultfile, "results.json.txt");
    logprintf(mystuff, "WARNING: Cannot read JSONResultsFile from INI file, using default (%s)\n", mystuff->jsonresultfile);
  }
  if (mystuff->verbosity >= 1)logprintf(mystuff, "  JSONResultsFile           %s\n", mystuff->jsonresultfile);

 /*****************************************************************************/

  if (my_read_string(mystuff->inifile, "LogFile", mystuff->logfile, 50))
  {
    sprintf(mystuff->logfile, "mfakto.log");
    logprintf(mystuff, "WARNING: Cannot read LogFile from INI file, using default (%s)\n", mystuff->logfile);
  }
  if (mystuff->verbosity >= 1)logprintf(mystuff, "  LogFile                   %s\n", mystuff->logfile);

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "Checkpoints", &i))
  {
    logprintf(mystuff, "WARNING: Cannot read Checkpoints from INI file, enabled by default\n");
    i=1;
  }
  else if(i < 0)
  {
    logprintf(mystuff, "WARNING: Checkpoints must be 0 (disabled) or greater, enabled by default\n");
    i=1;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i==0)logprintf(mystuff, "  Checkpoints               disabled\n");
    else if (i==1) logprintf(mystuff, "  Checkpoints               enabled\n");
    else           logprintf(mystuff, "  Checkpoints               every %d classes\n", i);
  }
  mystuff->checkpoints = i;

/*****************************************************************************/
  if (mystuff->checkpoints > 1)
  {
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  CheckpointDelay           ignored as Checkpoints > 1\n");
    mystuff->checkpointdelay = 300;
  }
  else
  {
    if(my_read_int(mystuff->inifile, "CheckpointDelay", &i))
    {
      logprintf(mystuff, "WARNING: Cannot read CheckpointDelay from INI file, set to 300 s by default\n");
      i = 300;
    }
    if(i > 3600)
    {
      logprintf(mystuff, "WARNING: Maximum value for CheckpointDelay is 3600 s\n");
      i = 3600;
    }
    if(i < 0)
    {
      logprintf(mystuff, "WARNING: Minimum value for CheckpointDelay is 0 s\n");
      i = 0;
    }
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  CheckpointDelay           %d s\n", i);
    mystuff->checkpointdelay = i;
  }

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "Stages", &i))
  {
    logprintf(mystuff, "WARNING: Cannot read Stages from INI file, enabled by default\n");
    i=1;
  }
  else if(i != 0 && i != 1)
  {
    logprintf(mystuff, "WARNING: Stages must be 0 or 1, enabled by default\n");
    i=1;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i==0)logprintf(mystuff, "  Stages                    disabled\n");
    else    logprintf(mystuff, "  Stages                    enabled\n");
  }
  mystuff->stages = i;

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "StopAfterFactor", &i))
  {
    logprintf(mystuff, "WARNING: Cannot read StopAfterFactor from INI file, set to 1 by default\n");
    i=1;
  }
  else if( (i < 0) || (i > 2) )
  {
    logprintf(mystuff, "WARNING: StopAfterFactor must be 0, 1 or 2, set to 1 by default\n");
    i=1;
  }
  if(mystuff->verbosity >= 1)
  {
         if(i==0)logprintf(mystuff, "  StopAfterFactor           disabled\n");
    else if(i==1)logprintf(mystuff, "  StopAfterFactor           bitlevel\n");
    else if(i==2)logprintf(mystuff, "  StopAfterFactor           class\n");
  }
  mystuff->stopafterfactor = i;

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "PrintMode", &i))
  {
    logprintf(mystuff, "WARNING: Cannot read PrintMode from INI file, set to 0 by default\n");
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    logprintf(mystuff, "WARNING: PrintMode must be 0 or 1, set to 0 by default\n");
    i=0;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i == 0)logprintf(mystuff, "  PrintMode                 full\n");
    else      logprintf(mystuff, "  PrintMode                 compact\n");
  }
  mystuff->printmode = i;

  /*****************************************************************************/


  if (my_read_int(mystuff->inifile, "Logging", &i))
  {
      logprintf(mystuff, "WARNING: Cannot read Logging from INI file, set to 0 by default\n");
      i = 0;
  }
  else if (i != 0 && i != 1)
  {
      logprintf(mystuff, "WARNING: Logging must be 0 or 1, set to 0 by default\n");
      i = 0;
  }
  if (mystuff->verbosity >= 1)
  {
      if (i == 0)logprintf(mystuff, "  Logging                   disabled\n");
      else      logprintf(mystuff, "  Logging                   enabled\n");
  }
  mystuff->logging = i;
  if (mystuff->logging == 1 && mystuff->logfileptr == NULL)
  {
      mystuff->logfileptr = fopen(mystuff->logfile, "a");
      if (mystuff->logfileptr == NULL)
      {
          logprintf(mystuff, "WARNING: Cannot open %s for appending, error: %d", mystuff->logfile, errno);
      }
  }

/*****************************************************************************/

  if (my_read_string(mystuff->inifile, "V5UserID", mystuff->V5UserID, 50))
  {
    /* no problem, don't use any */
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  V5UserID                  none\n");
    mystuff->V5UserID[0]='\0';
  }
  else
  {
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  V5UserID                  %s\n", mystuff->V5UserID);
  }

/*****************************************************************************/

  if(my_read_string(mystuff->inifile, "ComputerID", mystuff->ComputerID, 50))
  {
    /* no problem, don't use any */
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  ComputerID                none\n");
    mystuff->ComputerID[0]='\0';
  }
  else
  {
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  ComputerID                %s\n", mystuff->ComputerID);
  }

/*****************************************************************************/

  for(i = 0; i < 256; i++)mystuff->stats.progressheader[i] = 0;
  if(my_read_string(mystuff->inifile, "ProgressHeader", mystuff->stats.progressheader, 250))
  {
//    sprintf(mystuff->stats.progressheader, "    class | candidates |    time |    ETA | avg. rate | SievePrimes | CPU wait");
    sprintf(mystuff->stats.progressheader, "Date   Time     Pct    ETA | Exponent    Bits | GHz-d/day    Sieve     Wait");
    logprintf(mystuff, "WARNING: no ProgressHeader specified in INI file, using default\n");
  }
  if(mystuff->verbosity >= 2)logprintf(mystuff, "  ProgressHeader            \"%s\"\n", mystuff->stats.progressheader);

/*****************************************************************************/

  for(i = 0; i < 256; i++)mystuff->stats.progressformat[i] = 0;
  if(my_read_string(mystuff->inifile, "ProgressFormat", mystuff->stats.progressformat, 250))
  {
    sprintf(mystuff->stats.progressformat, "%%d %%T  %%p %%e | %%M %%l-%%u |   %%g  %%s  %%W%%%%");
    logprintf(mystuff, "WARNING: no ProgressFormat specified in INI file, using default\n");
  }
  if(mystuff->verbosity >= 2)logprintf(mystuff, "  ProgressFormat            \"%s\"\n", mystuff->stats.progressformat);


/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "TimeStampInResults", &i))
  {
    // no big deal, just leave it out
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    logprintf(mystuff, "WARNING: TimeStampInResults must be 0 or 1, set to 0 by default\n");
    i=0;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i == 0)logprintf(mystuff, "  TimeStampInResults        no\n");
    else      logprintf(mystuff, "  TimeStampInResults        yes\n");
  }
  mystuff->print_timestamp = i;

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "VectorSize", &i))
  {
    logprintf(mystuff, "WARNING: Cannot read VectorSize from INI file, set to 4 by default\n");
    i=4;
  }
  else if((i < 1 || i > 4) && i != 8 && i != 16 )
  {
    logprintf(mystuff, "WARNING: VectorSize must be one of 1, 2, 3, 4, or 8, set to 4 by default\n");
    i=4;
  }
#ifdef CHECKS_MODBASECASE
  if (i>1)
  {
    logprintf(mystuff, "WARNING: Reducing vector size from %d to 1 due to CHECKS_MODBASECASE.\n", i);
    i=1;
  }
#endif
  if(mystuff->verbosity >= 1)logprintf(mystuff, "  VectorSize                %d\n", i);
  mystuff->vectorsize = i;

/*****************************************************************************/

  if (my_read_string(mystuff->inifile, "GPUType", tmp, 50))
  {
    logprintf(mystuff, "WARNING: Cannot read GPUType from INI file, using default (AUTO)\n");
    strcpy(tmp, "AUTO");
    mystuff->gpu_type = GPU_AUTO;
  }
  else
  {
    mystuff->gpu_type = GPU_UNKNOWN;
    for (i=0; i < (int)GPU_UNKNOWN; i++)
    {
      if (strcasecmp(tmp, gpu_types[i].gpu_name) == 0)
      {
        mystuff->gpu_type = gpu_types[i].gpu_type;
        break;
      }
    }
    if (mystuff->gpu_type == GPU_UNKNOWN)
    {
      logprintf(mystuff, "WARNING: Unknown setting \"%s\" for GPUType, using default (AUTO)\n", tmp);
      strcpy(tmp, "AUTO");
      mystuff->gpu_type = GPU_AUTO;
    }
  }

  if(mystuff->verbosity >= 1)logprintf(mystuff, "  GPUType                   %s\n", tmp);

  /*****************************************************************************/

  if(my_read_int(mystuff->inifile, "SmallExp", &i))
  {
    logprintf(mystuff, "WARNING: Cannot read SmallExp from INI file, set to 0 by default\n");
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    logprintf(mystuff, "WARNING: SmallExp must be 0 or 1, set to 0 by default\n");
    i=0;
  }
  if(mystuff->verbosity >= 1)
  {
    if(i == 0)logprintf(mystuff, "  SmallExp                  no\n");
    else      logprintf(mystuff, "  SmallExp                  yes\n");
  }
  mystuff->small_exp = i;

  /*****************************************************************************/

  if(my_read_string(mystuff->inifile, "OCLCompileOptions", mystuff->CompileOptions, 150))
  {
    mystuff->CompileOptions[0]='\0';
  }
  else
  {
    if(mystuff->verbosity >= 1)logprintf(mystuff, "  Additional compile options %s\n", mystuff->CompileOptions);
  }

  /*****************************************************************************/

  if(my_read_string(mystuff->inifile, "UseBinfile", mystuff->binfile, 50))
  {
    mystuff->binfile[0] = '\0';
  }

  if(mystuff->verbosity >= 1)
  {
    logprintf(mystuff, "  UseBinfile                %s\n", mystuff->binfile);
  }

  /*****************************************************************************/
  return 0;
}

/* read a config array of integers from <filename>,
   search for keyname=,
   write at most num elements into arr,
   return the number of elements or -1 as error indicator */
int read_array(char *filename, char *keyname, cl_uint num, cl_uint *arr)
{
  FILE *in;
  char buf[512], tmp[512];
  char *ps, *pt, *pswap;
  cl_uint found = 0;
  cl_uint idx = (unsigned int) strlen(keyname);
  cl_uint i = 0;

  in=fopen(filename,"r");
  if(!in)
  {
    if (!inifile_unavailable)
    {
      char msg[80];
      inifile_unavailable = 1;
      sprintf(msg, "Cannot load INI file \"%.55s\"", filename);
      perror(msg);
    }
    return 1;
  }
  while(fgets(buf,512,in) && !found)
  {
    if(!strncmp(buf, keyname, idx) && buf[idx]=='=')
    {
      ps = buf + idx + 1;  // first char after =
      pt = tmp;
      for (i=0; i<num;)
      {
        found = sscanf(ps, "%u,%512s", &arr[i], pt);
        if (found) i++;
        if (found < 2 || *pt == '\0')  break;
        pswap = pt; pt = ps; ps = pswap;
        if (*ps == ',') ps++;
        if (*ps == ' ') ps++;
      }
      found = i;
    }
  }
  fclose(in);
  return i;
}
