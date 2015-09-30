/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2013  Oliver Weihe (o.weihe@t-online.de)
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
#include <limits.h>
#include <ctype.h>
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
  while(i*i <= n && i < 0x10000)
  {
    if(n%i==0)return 0;
    i+=2;
  }
  return 1;
}

int valid_assignment(unsigned int exp, int bit_min, int bit_max, int verbosity)
/*
returns 1 if the assignment is within the supported bounds of mfaktc,
0 otherwise.
*/
{
  int ret = 1;
  
       if(exp < 100000)       {ret = 0; if(verbosity >= 1)printf("WARNING: exponents < 100000 are not supported!\n");}
  else if(!isprime(exp))      {ret = 0; if(verbosity >= 1)printf("WARNING: exponent is not prime!\n");}
  else if(bit_min < 1 )       {ret = 0; if(verbosity >= 1)printf("WARNING: bit_min < 1 doesn't make sense!\n");}
  else if(bit_min > 94)       {ret = 0; if(verbosity >= 1)printf("WARNING: bit_min > 94 is not supported!\n");}
  else if(bit_min >= bit_max) {ret = 0; if(verbosity >= 1)printf("WARNING: bit_min >= bit_max doesn't make sense!\n");}
  else if(bit_max > 95)       {ret = 0; if(verbosity >= 1)printf("WARNING: bit_max > 95 is not supported!\n");}
  else if(((double)(bit_max-1) - (log((double)exp) / log(2.0F))) > 63.9F) /* this leave enough room so k_min/k_max won't overflow in tf_XX() */
                              {ret = 0; if(verbosity >= 1)printf("WARNING: k_max > 2^63.9 is not supported!\n");}
  
  if(verbosity >= 1 && ret == 0)printf("         Ignoring TF M%u from 2^%d to 2^%d!\n", exp, bit_min, bit_max);
  
  return ret;
}

enum PARSE_WARNINGS
{
  NO_WARNING=0,
  END_OF_FILE,
  LONG_LINE,
  NO_FACTOR_EQUAL,
  INVALID_FORMAT,
  INVALID_DATA,
  BLANK_LINE,
  NONBLANK_LINE
};

// note:  parse_worktodo_line() is a function that
//	returns the text of the line, the assignment data structure, and a success code.
enum PARSE_WARNINGS parse_worktodo_line(FILE *f_in, struct ASSIGNMENT *assignment, LINE_BUFFER *linecopy, char * *endptr)
/*
input
  f_in: an open file from where data is read
output
  assignment: structure of line, with any assignment if found
  linecopy: a copy of the last read line
  endptr: the end of data
*/
{
  char line[MAX_LINE_LENGTH+1], *ptr, *ptr_start, *ptr_end;
  int c;	// extended char pulled from stream;

  unsigned int scanpos;
  unsigned int number_of_commas;

  enum PARSE_WARNINGS reason = NO_WARNING;

  unsigned long proposed_exponent, proposed_bit_min, proposed_bit_max;

  if(NULL==fgets(line, MAX_LINE_LENGTH+1, f_in))
  {
    return END_OF_FILE;
  }
  if (linecopy != NULL)	// maybe it wasn't needed....
    strcpy(*linecopy,line);	// this is what was read...
  if((strlen(line) == MAX_LINE_LENGTH) && (!feof(f_in)) && (line[strlen(line)-1] !='\n') ) // long lines disallowed,
  {
    reason = LONG_LINE;
    do
    {
      c = fgetc(f_in);
      if ((EOF == c) ||(iscntrl(c)))	// found end of line
        break;
    }
    while(TRUE);
  }

  if (linecopy != NULL)
    *endptr = *linecopy;	// by default, non-significant content is whole line

  ptr=line;
  while (('\0'!=ptr[0]) && isspace(ptr[0]))	// skip leading spaces
    ptr++;
  if ('\0' == ptr[0])	// blank line...
    return BLANK_LINE;
  if( ('\\'==ptr[0]) && ('\\'==ptr[1]) )
    return NONBLANK_LINE;		// it's a comment, so ignore....don't care about long lines either..
  if( ('/'==ptr[0]) && ('/'==ptr[1]) )
    return NONBLANK_LINE;		// it's a comment, so ignore....don't care about long lines either..
  if (strncasecmp("Factor=", ptr, 7) != 0) // does the line start with "Factor="? (case-insensitive)
    return NO_FACTOR_EQUAL;
  ptr = 1+ strstr(ptr,"=");	// don't rescan..
  while (('\0'!=ptr[0]) && isspace(ptr[0]))	// ignore blanks...
    ptr++;
  number_of_commas = 0;
  for(scanpos = 0; scanpos < strlen(ptr); scanpos++)
  {
    if(ptr[scanpos] == ',')
      number_of_commas++; // count the number of ',' in the line
    if ((ptr[scanpos] == '\\') && (ptr[scanpos+1] == '\\'))
      break;	// comment delimiter
    if ((ptr[scanpos] == '/') && (ptr[scanpos+1] == '/'))
      break;	// //comment delimiter
  }
  if ((2!=number_of_commas) && (3!=number_of_commas))	// must have 2 or 3 commas...
    return INVALID_FORMAT;

  if(2==number_of_commas)
    assignment->assignment_key[0] = '\0';
  else
  {
    strncpy(assignment->assignment_key,ptr,1+(strstr(ptr,",")-ptr) );	// copy the comma..
    *strstr(assignment->assignment_key,",") = '\0';	// null-terminate key
    ptr=1 + strstr(ptr,",");
  }
  // ptr now points at exponent...in the future, the expression....
  ptr_start = ptr;
  while( (isspace(*ptr_start)) && ('\0' != *ptr_start ))
    ptr_start++;
  if ('M' == *ptr_start)	// M means Mersenne exponent...
    ptr_start++;
  errno = 0;
  proposed_exponent = strtoul(ptr_start, &ptr_end, 10);
  if (ptr_start == ptr_end)
    return INVALID_FORMAT;	// no conversion
  if ((0!=errno) || (proposed_exponent > UINT_MAX))
    return INVALID_DATA;	// for example, too many digits.
  ptr=ptr_end;

  // ptr now points at bit_min
  ptr_start = 1 + strstr(ptr,",");
  errno = 0;
  proposed_bit_min = strtoul(ptr_start, &ptr_end, 10);
  if (ptr_start == ptr_end)
    return INVALID_FORMAT;
  if ((0!=errno) || (proposed_bit_min > UCHAR_MAX))
    return INVALID_DATA;
  ptr = ptr_end;

  // ptr now points at bit_max
  ptr_start = 1 + strstr(ptr,",");
  errno =0;
  proposed_bit_max = strtoul(ptr_start, &ptr_end, 10);
  if (ptr_start == ptr_end)
    return INVALID_FORMAT;
  if ((0!=errno)||(proposed_bit_max > UCHAR_MAX) || (proposed_bit_max <= proposed_bit_min))
    return INVALID_DATA;
  ptr = ptr_end;
  while (('\0'!=ptr[0]) && isspace(ptr[0]))	// ignore blanks...
    ptr++;
  if (NULL != strstr(ptr,"\n"))		// kill off any trailing newlines...
    *strstr(ptr,"\n") = '\0';
  if (*ptr != '\0')
    strcpy(assignment->comment,ptr);

  if (linecopy != NULL)
    *endptr = *linecopy + (ptr_end - line);

  assignment->exponent = proposed_exponent;
  assignment->bit_min = proposed_bit_min;
  assignment->bit_max = proposed_bit_max;

  return reason;
}



/************************************************************************************************************
 * Function name : get_next_assignment                                                                      *
 *   													    *
 *     INPUT  :	char *filename										    *
 *		unsigned int *exponent									    *
 *		int *bit_min										    *
 *		int *bit_max										    *
 *		char *assignment_key[100];
 *     OUTPUT :                                        							    *
 *                                                                                                          *
 *     0 - OK												    *
 *     1 - get_next_assignment : cannot open file							    *
 *     2 - get_next_assignment : no valid assignment found						    *
 ************************************************************************************************************/
enum ASSIGNMENT_ERRORS get_next_assignment(char *filename, unsigned int *exponent, unsigned int *bit_min, unsigned int *bit_max, LINE_BUFFER *key, int verbosity)
{
  FILE *f_in;
  
  enum PARSE_WARNINGS value;
  struct ASSIGNMENT assignment;
  char *tail;
  LINE_BUFFER line;
  unsigned int linecount=0;

  // first, make sure we have an up-to-date worktodo file
  process_add_file(filename);
  f_in = fopen_and_lock(filename, "r");
  if(f_in == NULL)
  {
    printf("Can't open workfile %s\n", filename);
    return CANT_OPEN_FILE;	// nothing to open...
  }
  for(;;)
  {
    linecount++;
    value = parse_worktodo_line(f_in,&assignment,&line,&tail);
    if ((BLANK_LINE == value) || (NONBLANK_LINE == value))
      continue;
    if (NO_WARNING == value)
    {
      if (valid_assignment(assignment.exponent,assignment.bit_min, assignment.bit_max, verbosity))
        break;
      value = INVALID_DATA;
    }

    if (END_OF_FILE == value)
      break;
    if(verbosity >= 1)
    {
      printf("WARNING: ignoring line %u in \"%s\"! Reason: ", linecount, filename);
      switch(value)
      {
        case LONG_LINE:           printf("line is too long\n"); break;
        case NO_FACTOR_EQUAL:     printf("doesn't begin with Factor=\n");break;
        case INVALID_FORMAT:      printf("invalid format\n");break;
        case INVALID_DATA:        printf("invalid data\n");break;
        default:                  printf("unknown error on >%s<",line); break;
      }
    }
  }

  unlock_and_fclose(f_in);

  if (NO_WARNING == value)
  {
    *exponent = assignment.exponent;
    *bit_min = assignment.bit_min;
    *bit_max = assignment.bit_max;

    if (key!=NULL)strcpy(*key,assignment.assignment_key);

    return OK;
  }
  else
    return VALID_ASSIGNMENT_NOT_FOUND;
}


/************************************************************************************************************
 * Function name : clear_assignment                                                                         *
 *   													    *
 *     INPUT  :	char *filename										    *
 *		unsigned int exponent									    *
 *		int bit_min		- from old assignment file			    *
 *		int bit_max										    *
 *		int bit_min_new	- new bit_min,what was factored to--if 0,reached bit_max	    *
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
enum ASSIGNMENT_ERRORS clear_assignment(char *filename, unsigned int exponent, int bit_min, int bit_max, int bit_min_new)
{
  int found = FALSE;
  FILE *f_in, *f_out;
  LINE_BUFFER line;	// line buffer
  char *tail = NULL;	// points to tail material in line, if non-null
  enum PARSE_WARNINGS value;
  unsigned int line_to_drop = UINT_MAX;
  unsigned int current_line;
  struct ASSIGNMENT assignment;	// the found assignment....

  f_in = fopen_and_lock(filename, "r");
  if (NULL == f_in)
    return CANT_OPEN_WORKFILE;

  f_out = fopen_and_lock("__worktodo__.tmp", "w");
  if (NULL == f_out)
  {
    unlock_and_fclose(f_in);
    return CANT_OPEN_TEMPFILE;
  }

    if ((bit_min_new > bit_min) && (bit_min_new < bit_max))	// modify only
    line_to_drop = UINT_MAX;
  else
  {
    current_line =0;
    while (END_OF_FILE != (value = parse_worktodo_line(f_in,&assignment,&line,&tail)) )
    {
      current_line++;
      if (NO_WARNING == value)
      {
        if( (exponent == assignment.exponent) && (bit_min == assignment.bit_min) && (bit_max == assignment.bit_max))	// make final decision
        {
          if (line_to_drop > current_line)
          line_to_drop = current_line;
          break;
        }
        else
        {
          line_to_drop = current_line+1;	// found different assignment, can drop no earlier than next line
        }
      }
      else if ((BLANK_LINE == value) && (UINT_MAX == line_to_drop))
        line_to_drop = current_line+1;
    }
  }
  
  errno = 0;
  if (fseek(f_in,0L,SEEK_SET))
  {
    unlock_and_fclose(f_in);
    f_in = fopen_and_lock(filename, "r");
    if (NULL == f_in)
    {
      unlock_and_fclose(f_out);
      return CANT_OPEN_WORKFILE;
    }
  }
  
  found = FALSE;
  current_line = 0;
  while (END_OF_FILE != (value = parse_worktodo_line(f_in,&assignment,&line,&tail)) )
  {
    current_line++;
    if ((NO_WARNING != value) || found)
    {
      if ((found) || (current_line < line_to_drop))
        fprintf(f_out, "%s", line);
    }
    else	// assignment on the line, so we may need to print it..
    {
      found =( (exponent == assignment.exponent) && (bit_min == assignment.bit_min) && (bit_max == assignment.bit_max) );
      if (!found)
      {
        fprintf(f_out,"%s",line);
      }
      else	// we have the assignment...
      {
        if ((bit_min_new > bit_min) && (bit_min_new < bit_max))
        {
          fprintf(f_out,"Factor=" );
          if (strlen(assignment.assignment_key) != 0)
            fprintf(f_out,"%s,", assignment.assignment_key);
          fprintf(f_out,"%u,%u,%u",exponent, bit_min_new, bit_max);
          if (tail != NULL)
            fprintf(f_out,"%s",tail);
        }
      }
    }
  }	// while.....
  unlock_and_fclose(f_out);
  unlock_and_fclose(f_in);

  if (!found)
    return ASSIGNMENT_NOT_FOUND;
  if(remove(filename) != 0)
    return CANT_RENAME;
  if(rename("__worktodo__.tmp", filename) != 0)
    return CANT_RENAME;
  return OK;
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
      return 1;
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
