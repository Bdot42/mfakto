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
#include <string.h>
#if defined(BUILD_OPENCL)
#include <CL/cl.h>
#endif

#include "params.h"
#include "my_types.h"

extern kernel_info_t       kernel_info[];
extern struct GPU_type     gpu_types[];

static int my_read_int(char *inifile, char *name, int *value)
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

static int my_read_ulong(char *inifile, char *name, unsigned long long int *value)
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
      if(sscanf(&(buf[strlen(name)+1]),"%llu",value)==1)found=1;
    }
  }
  fclose(in);
  if(found)return 0;
  return 1;
}

static int my_read_string(char *inifile, char *name, char *string, unsigned int len)
{
  FILE *in;
  char buf[512];
  unsigned int found=0;
  unsigned int idx = (unsigned int) strlen(name);

  in=fopen(inifile,"r");
  if(!in)return 1;
  while(fgets(buf,512,in) && !found)
  {
    if(!strncmp(buf,name,idx) && buf[idx]=='=')
    {
      found = (unsigned int) strlen(buf + idx + 1);
      found = (len > found ? found : len) - 1;
      if (found)
        strncpy(string, buf+idx+1, found);
      string[found]='\0';
    }
  }
  fclose(in);
  if(found>0)return 0;
  return 1;
}

static int set_print_line(mystuff_t *mystuff)
{
  char printparms[]="CcpgtenrswWdTUHMlu";  /* list of allowed print parameters,
                                         when adding some, also update PRINT_PARM in my_types.h */
  char buf[512]={0};
  char *ppos;
  unsigned int i,out_pos=0,out_parm=0;

  for (i=0; i<strlen(printparms); i++)
  {
    mystuff->p_par[i].parm = printparms[i];
    mystuff->p_par[i].pos = 0;
  }

  for (i=0; i<strlen(mystuff->print_line) && out_pos<510; i++)  // 510 to allow writing 2 bytes plus \0 without checking again
  {
    buf[out_pos++] = mystuff->print_line[i];
    if (mystuff->print_line[i] == '%')
    {
      i++;
      ppos = strchr(printparms, mystuff->print_line[i]);
      if (ppos != NULL && mystuff->print_line[i]!='\0')
      {
        // found a valid param: remember the order number
        // special handling for strings already known right now (UserID and ComputerID)
        if (*ppos == 'U')
        {
          strncpy(buf+out_pos-1, mystuff->V5UserID, 512 - out_pos);
          out_pos = out_pos -1 + (unsigned int)strlen(mystuff->V5UserID);
          strncpy(mystuff->p_par[USER].out, mystuff->V5UserID, 15);
          mystuff->p_par[USER].out[15]='\0';
        }
        else if (*ppos == 'H')
        {
          strncpy(buf+out_pos-1, mystuff->ComputerID, 512 - out_pos);
          out_pos = out_pos -1 + (unsigned int)strlen(mystuff->ComputerID);
          strncpy(mystuff->p_par[HOST].out, mystuff->ComputerID, 15);
          mystuff->p_par[HOST].out[15]='\0';
        }
        else
        {
          if (out_parm >= 20) // too many params
          {
            fprintf(stderr, "Warning: More than 20 parameters in output format - line truncated\n");
            break;
          }
          // use '%s' in the format string
          buf[out_pos++] = 's';
          // p_ptr[0] ... p_ptr[n] will point to the correct parameter's string,
          // even allowing to use the same parm multiple times
          mystuff->p_ptr[out_parm++] = mystuff->p_par[ppos - printparms].out;
          // remember the posistion, but this is only used as a flag: if != 0, this parm need to be formatted
          mystuff->p_par[ppos - printparms].pos = out_parm; // starts counting at 1
        }
      }
      else
      {  
        buf[out_pos++] = '%';  // double the % sign to escape it
        i--;                   // start over with the char that did not match a known format.
                               // do not just copy it - it could be the leading %-sign of a known format
      }
    }
  }

  strncpy(mystuff->print_line, buf, 512);
  mystuff->print_line[511]='\0';
  return 0;
}

int read_config(mystuff_t *mystuff)
{
  int i;
  char tmp[51];
  unsigned long long int ul;

  printf("\nRuntime options\n");
  printf("  Inifile                   %s\n",mystuff->inifile);

/*****************************************************************************/  

  if(my_read_int(mystuff->inifile, "SievePrimesMin", &i))
  {
    printf("WARNING: Cannot read SievePrimesMin from inifile, using default value (%d)\n", 5000);
    i=5000;
  }
  else if((i < SIEVE_PRIMES_MIN) || (i >= SIEVE_PRIMES_MAX))
  {
    printf("WARNING: SievePrimesMin must be between %d and %d, using default value (%d)\n",
        SIEVE_PRIMES_MIN, SIEVE_PRIMES_MAX, 5000);
    i=5000;
  }
  printf("  SievePrimesMin            %d\n",i);
  mystuff->sieve_primes_min = i;

/*****************************************************************************/  

  if(my_read_int(mystuff->inifile, "SievePrimesMax", &i))
  {
    printf("WARNING: Cannot read SievePrimesMax from inifile, using default value (%d)\n", SIEVE_PRIMES_MAX);
    i=SIEVE_PRIMES_MAX;
  }
  else if((i < (int) mystuff->sieve_primes_min) || (i > SIEVE_PRIMES_MAX))
  {
    printf("WARNING: SievePrimesMax must be between SievePrimesMin(%d) and %d, using default value (%d)\n",
        mystuff->sieve_primes_min, SIEVE_PRIMES_MAX, 200000);
    i=200000;
  }
  printf("  SievePrimesMax            %d\n",i);
  mystuff->sieve_primes_max_global = i;

/*****************************************************************************/
  if(my_read_int(mystuff->inifile, "SievePrimes", &i))
  {
    printf("WARNING: Cannot read SievePrimes from inifile, using default value (%d)\n", SIEVE_PRIMES_DEFAULT);
    i=SIEVE_PRIMES_DEFAULT;
  }
  else
  {
    if((cl_uint)i>mystuff->sieve_primes_max_global)
    {
      printf("WARNING: Read SievePrimes=%d from inifile, using max value (%d)\n", i, mystuff->sieve_primes_max_global);
      i=mystuff->sieve_primes_max_global;
    }
    else if( i < (int) mystuff->sieve_primes_min)
    {
      printf("WARNING: Read SievePrimes=%d from inifile, using min value (%d)\n", i, mystuff->sieve_primes_min);
      i=mystuff->sieve_primes_min;
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

  if(my_read_string(mystuff->inifile, "WorkFile", mystuff->workfile, 50))
  {
    sprintf(mystuff->workfile, "worktodo.txt");
    printf("WARNING: Cannot read WorkFile from inifile, using default (%s)\n", mystuff->workfile);
  }
  printf("  WorkFile                  %s\n", mystuff->workfile);

/*****************************************************************************/

  if(my_read_string(mystuff->inifile, "ResultsFile", mystuff->resultsfile, 50))
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

  if (my_read_string(mystuff->inifile, "V5UserID", mystuff->V5UserID, 50))
  {
    /* no problem, don't use any */
    printf("  V5UserID                  none\n");
    mystuff->V5UserID[0]='\0';
  }
  else
  {
    printf("  V5UserID                  %s\n", mystuff->V5UserID);
  }

/*****************************************************************************/

  if(my_read_string(mystuff->inifile, "ComputerID", mystuff->ComputerID, 50))
  {
    /* no problem, don't use any */
    printf("  ComputerID                none\n");
    mystuff->ComputerID[0]='\0';
  }
  else
  {
    printf("  ComputerID                %s\n", mystuff->ComputerID);
  }

/*****************************************************************************/

  if(my_read_string(mystuff->inifile, "ProgressHeader", mystuff->head_line, 510))
  {
    /* no problem, use some default */
    strcpy(mystuff->head_line, "  done |    ETA |     GHz |time/class|    #FCs | avg. rate | SieveP. |CPU idle");
  }
  else
  {
// do not clutter the screen too much
//    printf("  ProgressHeader            %s\n", mystuff->head_line);
  }

/*****************************************************************************/

  if(my_read_string(mystuff->inifile, "PrintFormat", mystuff->print_line, 510))
  {
    /* no problem, use some default */
    strcpy(mystuff->print_line, "%p% | %e | %g |  %ts | %n | %rM/s | %s | %W%");
  }
  else
  {
// do not clutter the screen too much
//    printf("  PrintFormat               %s\n", mystuff->print_line);
  }
  set_print_line(mystuff);

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

  if(my_read_int(mystuff->inifile, "TimeStampInResults", &i))
  {
    // no big deal, just leave it out
    i=0;
  }
  else if(i != 0 && i != 1)
  {
    printf("WARNING: TimeStampInResults must be 0 or 1, set to 0 by default\n");
    i=0;
  }
  if(i == 0)printf("  TimeStampInResults        no\n");
  else      printf("  TimeStampInResults        yes\n");
  mystuff->print_timestamp = i;

/*****************************************************************************/

  if(my_read_int(mystuff->inifile, "VectorSize", &i))
  {
    printf("WARNING: Cannot read VectorSize from inifile, set to 4 by default\n");
    i=4;
  }
  else if((i < 1 || i > 4) && i != 8 && i != 16 )
  {
    printf("WARNING: VectorSize must be one of 1, 2, 3, 4, or 8, set to 4 by default\n");
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

  if (my_read_string(mystuff->inifile, "GPUType", tmp, 50))
  {
    printf("WARNING: Cannot read GPUType from inifile, using default (AUTO)\n");
    strcpy(tmp, "AUTO");
    mystuff->gpu_type = GPU_AUTO;
  }
  else
  {
    mystuff->gpu_type = GPU_UNKNOWN;
    for (i=0; i < (int)GPU_UNKNOWN; i++)
    {
      if (strcmp(tmp, gpu_types[i].gpu_name) == 0)
      {
        mystuff->gpu_type = gpu_types[i].gpu_type;
        break;
      }
    }
    if (mystuff->gpu_type == GPU_UNKNOWN)
    {
      printf("WARNING: Unknown setting \"%s\" for GPUType, using default (AUTO)\n", tmp);
      strcpy(tmp, "AUTO");
      mystuff->gpu_type = GPU_AUTO;
    }
  }

  printf("  GPUType                   %s\n", tmp);

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
  mystuff->gpu_sieving = i;

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

  if(my_read_ulong(mystuff->inifile, "SieveCPUMask", &ul))
  {
    printf("WARNING: Cannot read SieveCPUMask from inifile, set to 0 by default\n");
    ul=0;
  }
  printf("  SieveCPUMask              %lld\n", ul);
  
  mystuff->cpu_mask = ul;

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
  if(!in)return -1;
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