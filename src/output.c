/*
This file is part of mfaktc.
Copyright (C) 2009 - 2014  Oliver Weihe (o.weihe@t-online.de)
                           Bertram Franz (bertramf@gmx.net)

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
#include <math.h>
#include <time.h>
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#if defined __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif
#include "string.h"
#include "params.h"
#include "my_types.h"
#include "output.h"
#include "filelocking.h"
#include "compatibility.h"


void print_help(char *string)
{
  printf("mfakto (%s) Copyright (C) 2009-2014,\n", MFAKTO_VERSION);
  printf("  Oliver Weihe (o.weihe@t-online.de)\n");
  printf("  Bertram Franz (bertramf@gmx.net)\n");
  printf("This program comes with ABSOLUTELY NO WARRANTY; for details see COPYING.\n");
  printf("This is free software, and you are welcome to redistribute it\n");
  printf("under certain conditions; see COPYING for details.\n\n\n");

  printf("Usage: %s [options]\n", string);
  printf("  -h | --help            display this help\n");
  printf("  -d <xy>                specify to use OpenCL platform number x and\n");
  printf("                         device number y in this program\n");
  printf("  -d c                   force using all CPUs\n");
  printf("  -d g                   force using the first GPU\n");
  printf("  -v <n>                 verbosity level: 0=terse, 1=normal, 2=verbose, 3=debug\n");
  printf("  -tf <exp> <min> <max>  trial factor M<exp> from <min> to <max> bits\n");
  printf("                         instead of parsing the worktodo file\n");
  printf("  -i | --inifile <file>  load <file> as INI file (default: mfakto.ini)\n");
  printf("  -st                    self-test using the optimal kernel per test case\n");
  printf("  -st2                   self-test using all possible kernels\n");
  printf("\n");
  printf("options for debugging purposes\n");
  printf("  --timertest            test of timer functions\n");
  printf("  --sleeptest            test of sleep functions\n");
  printf("  --perftest [n]         performance tests, repeat each test <n> times (def: 10)\n");
  printf("  --CLtest               test of some OpenCL functions\n");
  printf("                         specify -d before --CLtest to test specified device\n");
}



void logprintf(mystuff_t* mystuff, const char* fmt, ...)
{
    va_list args;

    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);

    if (mystuff->logging == 1 && mystuff->logfileptr != NULL) {
        va_start(args, fmt);
        vfprintf(mystuff->logfileptr, fmt, args);
        va_end(args);
    }
}


/*
print_dezXXX(intXXX a, char *buf) writes "a" into "buf" in decimal
"buf" must be preallocated with enough space.
Enough space is
  23 bytes for print_dez72()  (2^72 -1  has 22 decimal digits)
  30 bytes for print_dez96()  (2^96 -1  has 29 decimal digits)
  45 bytes for print_dez144() (2^144 -1 has 44 decimal digits)
  59 bytes for print_dez192() (2^192 -1 has 58 decimal digits)

*/

void print_dez72(int96 a, char *buf)
{
  int96 tmp;

  tmp.d2 =                 a.d2 >> 16;
  tmp.d1 = (a.d2 << 16) + (a.d1 >>  8);
  tmp.d0 = (a.d1 << 24) +  a.d0;

  print_dez96(tmp, buf);
}


void print_dez90(int96 a, char *buf)
{
  int96 tmp;

  tmp.d2 =                 a.d2 >> 4;
  tmp.d1 = (a.d2 << 28) + (a.d1 >> 2);
  tmp.d0 = (a.d1 << 30) +  a.d0;

  print_dez96(tmp, buf);
}


void print_dez96(int96 a, char *buf)
{
  char digit[58];
  int digits=0,carry,i=0;
  cl_ulong tmp;

  while((a.d0!=0 || a.d1!=0 || a.d2!=0) && digits<58)
  {
                                              carry = a.d2%10; a.d2 /= 10;
    tmp = a.d1; tmp += (cl_ulong)carry << 32; carry = tmp%10;  a.d1 = (cl_uint) (tmp/10);
    tmp = a.d0; tmp += (cl_ulong)carry << 32; carry = tmp%10;  a.d0 = (cl_uint) (tmp/10);
    digit[digits++] = carry;
  }
  if(digits == 0)sprintf(buf, "0");
  else
  {
    digits--;
    while(digits >= 0)
    {
      buf[i++] = '0' + digit[digits--];
    }
    buf[i] = 0;
  }
}

/* unused

void print_dez144(int144 a, char *buf)
{
  int192 tmp;

  tmp.d5 = 0;
  tmp.d4 =                 a.d5 >>  8;
  tmp.d3 = (a.d5 << 24) +  a.d4;
  tmp.d2 = (a.d3 <<  8) + (a.d2 >> 16);
  tmp.d1 = (a.d2 << 16) + (a.d1 >>  8);
  tmp.d0 = (a.d1 << 24) +  a.d0;

  print_dez192(tmp, buf);
}


void print_dez192(int192 a, char *buf)
{
  char digit[58];
  int digits=0,carry,i=0;
  long long int tmp;

  while((a.d0!=0 || a.d1!=0 || a.d2!=0 || a.d3!=0 || a.d4!=0 || a.d5!=0) && digits<58)
  {
                                                   carry = a.d5%10; a.d5 /= 10;
    tmp = a.d4; tmp += (long long int)carry << 32; carry = tmp%10;  a.d4 = (cl_uint) (tmp/10);
    tmp = a.d3; tmp += (long long int)carry << 32; carry = tmp%10;  a.d3 = (cl_uint) (tmp/10);
    tmp = a.d2; tmp += (long long int)carry << 32; carry = tmp%10;  a.d2 = (cl_uint) (tmp/10);
    tmp = a.d1; tmp += (long long int)carry << 32; carry = tmp%10;  a.d1 = (cl_uint) (tmp/10);
    tmp = a.d0; tmp += (long long int)carry << 32; carry = tmp%10;  a.d0 = (cl_uint) (tmp/10);
    digit[digits++] = carry;
  }
  if(digits == 0)sprintf(buf, "0");
  else
  {
    digits--;
    while(digits >= 0)
    {
      sprintf(&(buf[i++]), "%1d", digit[digits--]);
    }
  }
}
*/

void print_timestamp(FILE *outfile)
{
  char* ptr;
  const time_t now = time(NULL);
  static time_t previous_time=0;

  if (previous_time + 5 < now) // have at least 5 seconds between successive time stamps in the results file
  {
    ptr = asctime(gmtime(&now));
    ptr[24] = '\0'; // cut off the newline
    fprintf(outfile, "[%s]\n", ptr);
    previous_time = now;
  }
}


void print_status_line(mystuff_t *mystuff)
{
  unsigned long long int eta;
  unsigned long long int elapsed;
  int i = 0, max_class_number;
  char buffer[256];
  int index = 0;
  time_t now;
  struct tm *tm_now = NULL;
  int time_read = 0;
  double val;

  if(mystuff->mode == MODE_SELFTEST_SHORT || mystuff->mode == MODE_PERFTEST) return; /* no output during short selftest */

  if (mystuff->more_classes)  max_class_number = 960;
  else                        max_class_number = 96;

  if(mystuff->stats.output_counter == 0)
  {
    logprintf(mystuff, "%s\n", mystuff->stats.progressheader);
    mystuff->stats.output_counter = 20;
  }
  if(mystuff->printmode == 0)mystuff->stats.output_counter--;

  while(mystuff->stats.progressformat[i] && i < 250)
  {
    if(mystuff->stats.progressformat[i] != '%')
    {
      buffer[index++] = mystuff->stats.progressformat[i];
      i++;
    }
    else
    {
      if(mystuff->stats.progressformat[i+1] == 'C')  // Class ID
      {
        index += sprintf(buffer + index, "%4d", mystuff->stats.class_number);
      }
      else if(mystuff->stats.progressformat[i+1] == 'c') // Class counter
      {
        index += sprintf(buffer + index, "%3d", mystuff->stats.class_counter);
      }
      else if(mystuff->stats.progressformat[i+1] == 'p') // percent complete
      {
        index += sprintf(buffer + index, "%5.1f", (double)(mystuff->stats.class_counter * 100) / (double)max_class_number);
      }
      else if(mystuff->stats.progressformat[i+1] == 'g') // speed (GHz-days/day)
      {
        if(mystuff->mode == MODE_NORMAL)
          index += sprintf(buffer + index, "%8.2f", mystuff->stats.ghzdays * 86400000.0f / ((double)mystuff->stats.class_time * (double)max_class_number));
        else
          index += sprintf(buffer + index, "   n.a.");
      }
      else if(mystuff->stats.progressformat[i+1] == 't') // time per class
      {
             if(mystuff->stats.class_time < 100000ULL  )index += sprintf(buffer + index, "%6.3f", (double)mystuff->stats.class_time/1000.0);
        else if(mystuff->stats.class_time < 1000000ULL )index += sprintf(buffer + index, "%6.2f", (double)mystuff->stats.class_time/1000.0);
        else if(mystuff->stats.class_time < 10000000ULL)index += sprintf(buffer + index, "%6.1f", (double)mystuff->stats.class_time/1000.0);
        else                                            index += sprintf(buffer + index, "%6.0f", (double)mystuff->stats.class_time/1000.0);
      }
      else if (mystuff->stats.progressformat[i + 1] == 'E')
      {
          if (mystuff->mode == MODE_NORMAL)
          {
              elapsed = mystuff->stats.bit_level_time / 1000;
                   if (elapsed < 3600) index += sprintf(buffer + index, "%2" PRIu64 "m%02" PRIu64 "s", elapsed / 60, elapsed % 60);
              else if (elapsed < 86400)index += sprintf(buffer + index, "%2" PRIu64 "h%02" PRIu64 "m", elapsed / 3600, (elapsed / 60) % 60);
              else                     index += sprintf(buffer + index, "%2" PRIu64 "d%02" PRIu64 "h", elapsed / 86400, (elapsed / 3600) % 24);
          }
          else if (mystuff->mode == MODE_SELFTEST_FULL)index += sprintf(buffer + index, "  n.a.");
      }
      else if(mystuff->stats.progressformat[i+1] == 'e') // eta
      {
        if(mystuff->mode == MODE_NORMAL)
        {
          eta = (mystuff->stats.class_time * (max_class_number - mystuff->stats.class_counter) + 500)  / 1000;
               if(eta < 3600) index += sprintf(buffer + index, "%2" PRIu64 "m%02" PRIu64 "s", eta / 60, eta % 60);
          else if(eta < 86400)index += sprintf(buffer + index, "%2" PRIu64 "h%02" PRIu64 "m", eta / 3600, (eta / 60) % 60);
          else                index += sprintf(buffer + index, "%2" PRIu64 "d%02" PRIu64 "h", eta / 86400, (eta / 3600) % 24);
        }
        else                  index += sprintf(buffer + index, "  n.a.");
      }
      else if(mystuff->stats.progressformat[i+1] == 'n') // number of candidates (CPU sieve: sieved number, GPU sieve: raw number)
      {
        if (mystuff->gpu_sieving == 1)
        {
          if(mystuff->stats.grid_count < (1000000000 / mystuff->gpu_sieve_processing_size + 1))
            index += sprintf(buffer + index, "%6.2fM", (double)mystuff->stats.grid_count * mystuff->gpu_sieve_processing_size / 1000000.0);
          else
            index += sprintf(buffer + index, "%6.2fG", (double)mystuff->stats.grid_count * mystuff->gpu_sieve_processing_size / 1000000000.0);
        } else {					// CPU sieving
          if(((unsigned long long int)mystuff->threads_per_grid * (unsigned long long int)mystuff->stats.grid_count) < 1000000000ULL)
            index += sprintf(buffer + index, "%6.2fM", (double)mystuff->threads_per_grid * (double)mystuff->stats.grid_count / 1000000.0);
          else
            index += sprintf(buffer + index, "%6.2fG", (double)mystuff->threads_per_grid * (double)mystuff->stats.grid_count / 1000000000.0);
        }
      }
      else if(mystuff->stats.progressformat[i+1] == 'r') // FC rate
      {
        if (mystuff->gpu_sieving == 1)
          val = (double)mystuff->stats.grid_count * mystuff->gpu_sieve_processing_size / ((double)mystuff->stats.class_time * 1000.0);
        else						// CPU sieving
          val = (double)mystuff->threads_per_grid * (double)mystuff->stats.grid_count / ((double)mystuff->stats.class_time * 1000.0);

        if(val <= 999.99f) index += sprintf(buffer + index, "%6.2f", val);
        else               index += sprintf(buffer + index, "%6.1f", val);
      }
      else if(mystuff->stats.progressformat[i+1] == 's') // (GPU-)SievePrimes
      {
        index += sprintf(buffer + index, "%7d", mystuff->sieve_primes);
      }
      else if(mystuff->stats.progressformat[i+1] == 'w') // CPU wait time
      {
        index += sprintf(buffer + index, "%6u", (unsigned int) (mystuff->stats.cpu_wait_time / mystuff->stats.grid_count)); /* mfakto only */
      }
      else if(mystuff->stats.progressformat[i+1] == 'W') // CPU wait fraction
      {
        if(mystuff->stats.cpu_wait >= 0.0f)index += sprintf(buffer + index, "%6.2f", mystuff->stats.cpu_wait);
        else                               index += sprintf(buffer + index, "  n.a.");
      }
      else if(mystuff->stats.progressformat[i+1] == 'd') // date
      {
        if(!time_read)
        {
          now = time(NULL);
          tm_now = localtime(&now);
          time_read = 1;
        }
        index += (int)strftime(buffer + index, 7, "%b %d", tm_now);
      }
      else if(mystuff->stats.progressformat[i+1] == 'T') // time
      {
        if(!time_read)
        {
          now = time(NULL);
          tm_now = localtime(&now);
          time_read = 1;
        }
        index += (int)strftime(buffer + index, 6, "%H:%M", tm_now);
      }
      else if(mystuff->stats.progressformat[i+1] == 'U') // user
      {
        index += sprintf(buffer + index, "%s", mystuff->V5UserID);
      }
      else if(mystuff->stats.progressformat[i+1] == 'H') // host
      {
        index += sprintf(buffer + index, "%s", mystuff->ComputerID);
      }
      else if(mystuff->stats.progressformat[i+1] == 'M') // exponent
      {
        index += sprintf(buffer + index, "%-10u", mystuff->exponent);
      }
      else if(mystuff->stats.progressformat[i+1] == 'l') // low bit limit
      {
        index += sprintf(buffer + index, "%2d", mystuff->bit_min);
      }
      else if(mystuff->stats.progressformat[i+1] == 'u') // upper bit limit for this stage
      {
        index += sprintf(buffer + index, "%2d", mystuff->bit_max_stage);
      }
      else if(mystuff->stats.progressformat[i+1] == '%')
      {
        buffer[index++] = '%';
      }
      else /* '%' + unknown format character -> just print "%<character>" */
      {
        buffer[index++] = '%';
        --i; // advance i only by 1, with the +=2 below. This way, we don't run over the end of the string if the last char is '%'
      }

      i += 2;
    }
    if(index > 200) /* buffer has 256 bytes, single format strings are limited to 50 bytes */
    {
      buffer[index] = 0;
      logprintf(mystuff, "%s", buffer);
      index = 0;
    }
  }


  if(mystuff->mode == MODE_NORMAL)
  {
    if(mystuff->printmode == 1)index += sprintf(buffer + index, "\r");
    else                       index += sprintf(buffer + index, "\n");
  }

  if((mystuff->mode > MODE_SELFTEST_SHORT) && (mystuff->printmode == 0))
  {
    index += sprintf(buffer + index, "\n");
  }

  buffer[index] = 0;
  logprintf(mystuff, "%s", buffer);
}

void get_utc_timestamp(char* timestamp)
{
    time_t now;
    struct tm* utc_time;

    time(&now);
    utc_time = gmtime(&now);
    strftime(timestamp, sizeof(char[50]), "%Y-%m-%d %H:%M:%S", utc_time);
}

const char* getArchitectureJSON() {
#if defined(__x86_64__) || defined(_M_X64)
    return ", \"architecture\": \"x86_64\"";
#elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
    return ", \"architecture\": \"x86_32\"";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return ", \"architecture\": \"ARM64\"";
#else
    return "";
#endif
}

void getOSJSON(char* string) {
#if defined(_WIN32) || defined(_WIN64)
    sprintf(string, ", \"os\":{\"os\": \"Windows\"%s}", getArchitectureJSON());
#elif defined(__APPLE__)
    sprintf(string, ", \"os\":{\"os\": \"Darwin\"%s}", getArchitectureJSON());
#elif defined(__linux__)
    sprintf(string, ", \"os\":{\"os\": \"Linux\"%s}", getArchitectureJSON());
#elif defined(__unix__)
    sprintf(string, ", \"os\":{\"os\": \"Unix\"%s}", getArchitectureJSON());
#endif
}


void print_result_line(mystuff_t *mystuff, int factorsfound)
/* printf the final result line (STDOUT and resultfile) */
{
  char UID[110]; /* 50 (V5UserID) + 50 (ComputerID) + 8 + spare */
  char aidjson[111];
  char userjson[61]; /* 50 (V5UserID) + 11 spare */
  char computerjson[65];  /* 50 (ComputerID) + 15 spare */
  char factorjson[513];
  char osjson[200];
  char txtstring[200];
  char timestamp[50];

  unsigned int max_class_number;

  FILE *txtresultfile=NULL;

  char jsonstring[1100];
  FILE *jsonresultfile=NULL;

  if (mystuff->more_classes)  max_class_number = 960;
  else                        max_class_number = 96;

  if(mystuff->V5UserID[0] && mystuff->ComputerID[0])
    sprintf(UID, "UID: %s/%s, ", mystuff->V5UserID, mystuff->ComputerID);
  else
    UID[0]=0;

  if (mystuff->assignment_key[0] && strspn(mystuff->assignment_key, "0123456789abcdefABCDEF") == 32 && strlen(mystuff->assignment_key) == 32)
    sprintf(aidjson, ", \"aid\":\"%s\"", mystuff->assignment_key);
  else
    aidjson[0] = 0;

  if (mystuff->V5UserID[0])
    sprintf(userjson, ", \"user\":\"%s\"", mystuff->V5UserID);
  else
    userjson[0] = 0;

  if (mystuff->ComputerID[0])
    sprintf(computerjson, ", \"computer\":\"%s\"", mystuff->ComputerID);
  else
    computerjson[0] = 0;

  if (mystuff->factors_string[0])
    sprintf(factorjson, ", \"factors\":[%s]", mystuff->factors_string);
  else
    factorjson[0] = 0;

  getOSJSON(osjson);
  get_utc_timestamp(timestamp);

  if(mystuff->mode == MODE_NORMAL)
  {
    txtresultfile = fopen_and_lock(mystuff->resultfile, "a");
    if(mystuff->print_timestamp == 1)print_timestamp(txtresultfile);
    jsonresultfile = fopen_and_lock(mystuff->jsonresultfile, "a");
  }
  bool partialresult = (mystuff->mode == MODE_NORMAL) && (mystuff->stats.class_counter < max_class_number);
  if(factorsfound)
  {
      sprintf(txtstring, "found %d factor%s for M%u from 2^%2d to 2^%2d %s[%s %s]",
         factorsfound, (factorsfound > 1) ? "s" : "", mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage,
         partialresult ? "(partially tested) " : "", MFAKTO_VERSION, mystuff->stats.kernelname);
  }
  else
  {
    sprintf(txtstring, "no factor for M%u from 2^%d to 2^%d [%s %s]",
      mystuff->exponent, mystuff->bit_min, mystuff->bit_max_stage,
      MFAKTO_VERSION, mystuff->stats.kernelname);
  }
  sprintf(jsonstring, "{\"exponent\":%u, \"worktype\":\"TF\", \"status\":\"%s\", \"bitlo\":%2d, \"bithi\":%2d, \"rangecomplete\":%s%s, \"program\":{\"name\":\"mfakto\", \"version\":\"%s\", \"subversion\":\"%s\"}, \"timestamp\":\"%s\"%s%s%s%s}",
      mystuff->exponent, factorsfound > 0 ? "F" : "NF", mystuff->bit_min, mystuff->bit_max_stage, partialresult ? "false" : "true", factorjson, SHORT_MFAKTO_VERSION, mystuff->stats.kernelname, timestamp, userjson, computerjson, aidjson, osjson);

  if(mystuff->mode != MODE_SELFTEST_SHORT)
  {
    printf("%s\n", txtstring);
  }
  if(mystuff->mode == MODE_NORMAL)
  {
    fprintf(txtresultfile, "%s%s\n", UID, txtstring);
    unlock_and_fclose(txtresultfile);
    fprintf(jsonresultfile, "%s\n", jsonstring);
    unlock_and_fclose(jsonresultfile);
  }
}


void print_factor(mystuff_t *mystuff, int factor_number, char *factor, double bits)
{
  char UID[110]; /* 50 (V5UserID) + 50 (ComputerID) + 8 + spare */
  FILE *resultfile = NULL;
  unsigned int max_class_number;

  if (mystuff->more_classes)  max_class_number = 960;
  else                        max_class_number = 96;

  if(mystuff->V5UserID[0] || mystuff->ComputerID[0])
    sprintf(UID, "UID: %s/%s, ", mystuff->V5UserID, mystuff->ComputerID);
  else
    UID[0]=0;

  if(mystuff->mode == MODE_NORMAL)
  {
    resultfile = fopen_and_lock(mystuff->resultfile, "a");
    if(mystuff->print_timestamp == 1 && factor_number == 0)print_timestamp(resultfile);
  }

  if(factor_number < 10)
  {
    if(mystuff->mode != MODE_SELFTEST_SHORT)
    {
      if(mystuff->printmode == 1 && factor_number == 0)logprintf(mystuff, "\n");
      logprintf(mystuff, "M%u has a factor: %s (%f bits, %f GHz-d)\n", mystuff->exponent, factor, bits, mystuff->stats.ghzdays);
    }
    if(mystuff->mode == MODE_NORMAL)
    {
      fprintf(resultfile, "%sM%u has a factor: %s [TF:%d:%d%s:%s %s]\n",
        UID, mystuff->exponent, factor, mystuff->bit_min, mystuff->bit_max_stage,
        ((mystuff->stopafterfactor == 2) && (mystuff->stats.class_counter < max_class_number)) ? "*" : "" ,
        MFAKTO_VERSION, mystuff->stats.kernelname);
    }
  }
  else /* factor_number >= 10 */
  {
    if(mystuff->mode != MODE_SELFTEST_SHORT)      printf("M%u: %d additional factors not shown\n",      mystuff->exponent, factor_number-10);
    if(mystuff->mode == MODE_NORMAL)fprintf(resultfile,"%sM%u: %d additional factors not shown\n", UID, mystuff->exponent, factor_number-10);
  }

  if(mystuff->mode == MODE_NORMAL)unlock_and_fclose(resultfile);
}


double primenet_ghzdays(unsigned int exponent, int bit_min, int bit_max)
/* estimate the GHZ-days for the current job
GHz-days = <magic constant> * pow(2, $bitlevel - 48) * 1680 / $exponent

magic constant is 0.016968 for TF to 65-bit and above
magic constant is 0.017832 for 63-and 64-bit
magic constant is 0.011160 for 62-bit and below

example using M50,000,000 from 2^69-2^70:
 = 0.016968 * pow(2, 70 - 48) * 1680 / 50000000
 = 2.3912767291392 GHz-days*/
{
  // just use the 65-bit constant, that's close enough
  return 0.016968 * (double)(1ULL << (bit_min - 47)) * 1680 / exponent * ((1 << (bit_max-bit_min)) -1);
}

const char* ClErrorString( const cl_int errcode )
{
  switch ( errcode )
  {
    case CL_SUCCESS:                            // 0
      return "Success";
    case CL_DEVICE_NOT_FOUND:                   // -1
      return "Device not found";
    case CL_DEVICE_NOT_AVAILABLE:               // -2
      return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:             // -3
      return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      // -4
      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:                   // -5
      return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:                 // -6
      return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       // -7
      return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                   // -8
      return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:              // -9
      return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         // -10
      return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:              // -11
      return "Build program failure";
    case CL_MAP_FAILURE:                        // -12
      return "Map failure";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:       // -13
      return "Misaligned sub-buffer offset";
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: // -14
      return "Exec status error for events in wait list";
    case CL_COMPILE_PROGRAM_FAILURE:            // -15
      return "Compile program failure";
    case CL_LINKER_NOT_AVAILABLE:               // -16
      return "Linker not available";
    case CL_LINK_PROGRAM_FAILURE:               // -17
      return "Link program failure";
    case CL_DEVICE_PARTITION_FAILED:            // -18
      return "Device partition failed";
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      // -19
      return "Kernel argument info not available";


    case CL_INVALID_VALUE:                      // -30
      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:                // -31
      return "Invalid device type";
    case CL_INVALID_PLATFORM:                   // -32
      return "Invalid platform";
    case CL_INVALID_DEVICE:                     // -33
      return "Invalid device";
    case CL_INVALID_CONTEXT:                    // -34
      return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:           // -35
      return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:              // -36
      return "Invalid command queue";
    case CL_INVALID_HOST_PTR:                   // -37
      return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:                 // -38
      return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    // -39
      return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:                 // -40
      return "Invalid image size";
    case CL_INVALID_SAMPLER:                    // -41
      return "Invalid sampler";
    case CL_INVALID_BINARY:                     // -42
      return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:              // -43
      return "Invalid build options";
    case CL_INVALID_PROGRAM:                    // -44
      return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:         // -45
      return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:                // -46
      return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:          // -47
      return "Invalid kernel definition";
    case CL_INVALID_KERNEL:                     // -48
      return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                  // -49
      return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                  // -50
      return "Invalid argument size";
    case CL_INVALID_ARG_SIZE:                   // -51
      return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:                // -52
      return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:             // -53
      return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:            // -54
      return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:             // -55
      return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:              // -56
      return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:            // -57
      return "Invalid event wait list";
    case CL_INVALID_EVENT:                      // -58
      return "Invalid event";
    case CL_INVALID_OPERATION:                  // -59
      return "Invalid operation";
    case CL_INVALID_GL_OBJECT:                  // -60
      return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:                // -61
      return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                  // -62
      return "Invalid miplevel";
    case CL_INVALID_GLOBAL_WORK_SIZE:           // -63
      return "Invalid global work size";
    case CL_INVALID_PROPERTY:                   // -64
      return "Invalid property";
    case CL_INVALID_IMAGE_DESCRIPTOR:           // -65
      return "Invalid image descriptor";
    case CL_INVALID_COMPILER_OPTIONS:           // -66
      return "Invalid compiler options";
    case CL_INVALID_LINKER_OPTIONS:             // -67
      return "Invalid linker options";
    case CL_INVALID_DEVICE_PARTITION_COUNT:     // -68
      return "Invalid device partition count";


    case RET_ERROR:                             // 1000000001
      return "Internal mfakto error";
    case RET_QUIT:                              // 1000000002
      return "Exit due to Ctrl-C or signal";
    default:
      return "Unknown errorcode (not an OpenCL error)";
  }
}

void printArray(const char * Name, const cl_uint * Data, const cl_uint len, cl_uint hex)
{
  cl_uint i, o, c, val;
  char *fmt1, *fmt2, *fmt3, *fmt4;

  if (hex)
  {
    fmt1=(char *)"<%u x %#x> ";
    fmt2=(char *)"%#x ";
    fmt3=(char *)"... %#x %#x %#x\n";
    fmt4=(char *)"<%d x 0x0 at the end>\n";
  }
  else
  {
    fmt1=(char *)"<%u x %u> ";
    fmt2=(char *)"%u ";
    fmt3=(char *)"... %u %u %u\n";
    fmt4=(char *)"<%d x 0 at the end>\n";
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
