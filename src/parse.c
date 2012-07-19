/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2011  Oliver Weihe (o.weihe@t-online.de)
This file has been written by Luigi Morelli (L.Morelli@mclink.it) *1

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

/*
*1 Luigi initially wrote the two functions get_next_assignment() and
clear_assignment() after we (Luigi and myself (Oliver)) have discussed the
interface. Luigi was so nice to write those functions so I had time to focus
on other parts, this made early (mfaktc 0.07) worktodo parsing posible.
For mfakc 0.15 I've completly rewritten those two functions. The rewritten
functions should be more robust against malformed input. Grab the sources of
mfaktc 0.07-0.14 to see Luigis code.
*/


/************************************************************************************************************
 * Input/output file function library                                                                       *
 *                                                   							    *
 *   return codes:											    *
 *     0 - OK												    *
 *     1 - get_next_assignment : cannot open file							    *
 *     2 - get_next_assignment : no valid assignment found						    *
 *     3 - clear_assignment    : cannot open file <filename>						    *
 *     4 - clear_assignment    : cannot open file "__worktodo__.tmp"					    *
 *     5 - clear_assignment    : assignment not found							    *
 *     6 - clear_assignment    : cannot rename temporary workfile to regular workfile			    *
 ************************************************************************************************************/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>
#include "compatibility.h"
#include "filelocking.h"
#include "parse.h"

static int add_file_disabled=0;

int isprime(unsigned int n)
/*
returns
0 if n is composite
1 if n is prime
*/
{
  unsigned int i;
  
  if(n<=1) return 0;
  if(n>2 && n%2==0)return 0;

  i=3;
  while(i*i <= n)
  {
    if(n%i==0)return 0;
    i+=2;
  }
  return 1;
}

int valid_assignment(unsigned int exp, int bit_min, int bit_max)
/*
returns 1 if the assignment is within the supported bounds of mfakto,
0 otherwise.
*/
{
  int ret = 1;
  
       if(exp < 1000000)      {printf("WARNING: exponents < 1000000 are not supported!\n"); ret = 0;}
  else if(!isprime(exp))      {printf("WARNING: exponent is not prime!\n"); ret = 0;}
  else if(bit_min < 1 )       {printf("WARNING: bit_min < 1 doesn't make sense!\n"); ret = 0;}
  else if(bit_min > 94)       {printf("WARNING: bit_min > 94 is not supported!\n"); ret = 0;}
  else if(bit_min >= bit_max) {printf("WARNING: bit_min >= bit_max doesn't make sense!\n"); ret = 0;}
  else if(bit_max > 95)       {printf("WARNING: bit_max > 95 is not supported!\n"); ret = 0;}
  else if(((double)(bit_max-1) - (log((double)exp) / log(2.0F))) > 63.9F) /* this leave enough room so k_min/k_max won't overflow in tf_XX() */
                              {printf("WARNING: k_max > 2^63.9 is not supported!\n"); ret = 0;}                              
  
  if(ret == 0)printf("         Ignoring TF M%u from 2^%d to 2^%d!\n", exp, bit_min, bit_max);
  
  return ret;
}


/************************************************************************************************************
 * Function name : get_next_assignment                                                                      *
 *   													    *
 *     INPUT  :	char *filename										    *
 *		unsigned int *exponent									    *
 *		int *bit_min										    *
 *		int *bit_max										    *
 *     OUTPUT :                                        							    *
 *                                                                                                          *
 *     0 - OK												    *
 *     1 - get_next_assignment : cannot open file							    *
 *     2 - get_next_assignment : no valid assignment found						    *
 ************************************************************************************************************/
int get_next_assignment(char *filename, unsigned int *exponent, int *bit_min, int *bit_max)
{
  char line[101], *ptr, *ptr_start, *ptr_end, buf[50];
  int ret = 2, i,j, count = 0, reason = 0;
  FILE *f_in;

  // first, make sure we have an up-to-date worktodo file
  process_add_file(filename);
  f_in = fopen_and_lock(filename, "r");
  if(f_in != NULL)
  {
    while(fgets(line, 101, f_in) != NULL)
    {
      count ++;
      if(line[strlen(line) - 1] == '\n')line[strlen(line) - 1] = 0;

      if(strlen(line) == 100) // lets ignore long lines!
      {
        while(fgets(line, 101, f_in) != NULL && strlen(line) == 100);
        reason = 1;
      }
      else // not a long line
      {
        reason = 2;
        if(strncasecmp("Factor=", line, 7) == 0) // does the line start with "Factor="? (case-insensitive)
        {
          reason = 3;
          j = 0;
          for(i = 0; i < strlen(line); i++)if(line[i] == ',')j++; // count the number of ',' in the line
          if(j == 2 || j == 3) // only accept lines with 2 or 3 ','
          {
            if(j == 2)ptr = strstr(line, "="); // assume Factor=<exp>,<bit_min>,<bit_max>
            else      ptr = strstr(line, ","); // assume Factor=<some text>,<exp>,<bit_min>,<bit_max>
            ptr++;
            ptr_start = ptr;
            *exponent = strtoul(ptr_start, &ptr_end, 10);
            if(ptr_start != ptr_end && ptr_end[0] == ',')
            {
              ptr_start = ptr_end + 1;
              *bit_min = strtoul(ptr_start, &ptr_end, 10);
              if(ptr_start != ptr_end && ptr_end[0] == ',')
              {
                ptr_start = ptr_end + 1;
                *bit_max = strtoul(ptr_start, &ptr_end, 10);
                if(ptr_start != ptr_end)
                {
                  reason = 4;
                  sprintf(buf, "%u,%d,%d", *exponent, *bit_min, *bit_max);
                  if(strcmp(ptr, buf) == 0) // can we find what we've read?
                  {
                    reason = 0;
                    if(valid_assignment(*exponent, *bit_min, *bit_max))
                    {
                      unlock_and_fclose(f_in);
                      return 0;
                    }
                  }
                }
              }
            }
          }
        }
      }
      if(reason != 0) printf("WARNING: ignoring line %d in \"%s\"! Reason: ", count, filename);
           if(reason == 1) printf("line is too long\n");
      else if(reason == 2) printf("doesn't begin with Factor=\n");
      else if(reason == 3) printf("invalid format\n");
      else if(reason == 4) printf("invalid data\n");
    }
    unlock_and_fclose(f_in);
  }
  else // f_in == NULL
  {
    ret = 1; // Can't open file!
  }
  return ret;
}



/************************************************************************************************************
 * Function name : clear_assignment                                                                         *
 *   													    *
 *     INPUT  :	char *filename										    *
 *		unsigned int exponent									    *
 *		int bit_min										    *
 *		int bit_max										    *
 *		int bit_min_new										    *
 *     OUTPUT :                                        							    *
 *                                                                                                          *
 *     0 - OK												    *
 *     3 - clear_assignment    : cannot open file <filename>						    *
 *     4 - clear_assignment    : cannot open file "__worktodo__.tmp"					    *
 *     5 - clear_assignment    : assignment not found							    *
 *     6 - clear_assignment    : cannot rename temporary workfile to regular workfile			    *
 *                                                                                                          *
 * If bit_min_new is zero then the specified assignment will be cleared. If bit_min_new is greater than     *
 * zero the specified assignment will be modified                                                           *
 ************************************************************************************************************/

int clear_assignment(char *filename, unsigned int exponent, int bit_min, int bit_max, int bit_min_new)
{
  int ret = 5, i, j, found;
  FILE *f_in, *f_out;
  char line[101], *ptr, buf[50];
  
  f_in = fopen_and_lock(filename, "r");
  if(f_in != NULL)
  {
    f_out = fopen_and_lock("__worktodo__.tmp", "w");
    if(f_out != NULL)
    {
      while(fgets(line, 101, f_in) != NULL)
      {
        if(line[strlen(line) - 1] == '\n')line[strlen(line) - 1] = 0;
        if(strlen(line) == 100) // lets ignore long lines, just copy them
        {
          fprintf(f_out, "%s", line);
          while(fgets(line, 101, f_in) != NULL && strlen(line) == 100)fprintf(f_out, "%s", line);
          fprintf(f_out, "%s", line);
        }
        else // not a long line
        {
          found = 0;
          if(strncasecmp("Factor=", line, 7) == 0) // does the line start with "Factor="? (case-insensitive)
          {
            j = 0;
            for(i = 0; i < strlen(line); i++)if(line[i] == ',')j++; // count the number of ',' in the line
            if(j == 2 || j == 3) // only accept lines with 2 or 3 ','
            {
              if(j == 2)ptr = strstr(line, "="); // assume Factor=<exp>,<bit_min>,<bit_max>
              else      ptr = strstr(line, ","); // assume Factor=<some text>,<exp>,<bit_min>,<bit_max>
              ptr++;
              sprintf(buf, "%u,%d,%d", exponent, bit_min, bit_max);
              if(strcmp(ptr, buf) == 0)
              {
                found = 1;
                ret = 0;
                ptr[0] = 0;
                if(bit_min_new > 0)
                {
                  fprintf(f_out, "%s%u,%d,%d\n", line, exponent, bit_min_new, bit_max);
                }
              }
            }
          }
          if(found == 0)
          {
            fprintf(f_out, "%s\n", line);
          }
        }
      }
      unlock_and_fclose(f_out);
      unlock_and_fclose(f_in);
      if(remove(filename) != 0) ret = 6;
      else
      {
        if(rename("__worktodo__.tmp", filename) != 0) ret = 6;
      }
    }
    else // f_out == NULL
    {
      unlock_and_fclose(f_in);
      ret = 4;
    }
  }
  else // f_in == NULL
  {
    ret = 3;
  }
  
  return ret;
}


/* is there an add file for the worktodo file <filename> available ?
   ret == 1 : yes
   ret == 0 : no
 */
int add_file_available(char *filename)
{
	char	add_filename[256];
	char	*dot;

  if (add_file_disabled) return 0;  // there was an error with this add file earlier, ignore it forever

	strncpy (add_filename, filename, 245);  // leave room if ".add.txt" will be appended
  add_filename[245]='\0';
	dot = strrchr (add_filename, '.');
	if (dot == NULL)
  {
    dot = add_filename + strlen(add_filename);  // no dot? just append the extension
  }
	strcpy (dot, ".add");
	if (file_exists (add_filename)) return 1;
	strcpy (dot, ".add.txt");
	if (file_exists (add_filename)) return 1;

	return 0;
}

/* process the add file for the worktodo file <filename> */
int process_add_file(char *filename)
{
	char	add_filename[256];
	char	*dot;
  FILE  *f_work, *f_add;
  char  line[101];

  if (add_file_disabled) return 1;  // there was an error with this add file earlier
	strncpy (add_filename, filename, 245);  // leave room if ".add.txt" will be appended
  add_filename[245]='\0';
	dot = strrchr (add_filename, '.');
	if (dot == NULL)
  {
    dot = add_filename + strlen(add_filename);  // no dot? just append the extension
  }
	strcpy (dot, ".add");
	if (!file_exists (add_filename))
  {
	  strcpy (dot, ".add.txt");
	  if (!file_exists (add_filename)) return 0;  // no problem if there is no .add file
  }

  // here, add_filename contains an existing add file's name
  f_work = fopen_and_lock(filename, "a+");
  if (f_work == NULL) return 1;
  f_add = fopen_and_lock(add_filename, "r");
  if (f_add == NULL)
  {
    unlock_and_fclose(f_work);
    return 1;
  }

  printf("\nAdding \"%s\" to \"%s\".\n", add_filename, filename);
  while(fgets(line, 101, f_add) != NULL)
  {
    if (fputs(line, f_work) == EOF)
    {
      fprintf(stderr, "Error %d appending \"%s\" to \"%s\"\n", errno, add_filename, filename);
      add_file_disabled = 1;  // Do not try again in order to avoid duplicating entries
      break;
    }
  }
  unlock_and_fclose(f_add);
  unlock_and_fclose(f_work);
  if (remove(add_filename)!= 0)
  {
    perror("Failed to delete add_file");
    add_file_disabled = 1;  // Do not try again in order to avoid duplicating entries
  }

  return add_file_disabled;
}
