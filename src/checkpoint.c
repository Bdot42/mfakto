/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009, 2010, 2011  Oliver Weihe (o.weihe@t-online.de)

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

#include "params.h"

unsigned int checkpoint_checksum(char *string, int chars)
/* generates a CRC-32 like checksum of the string */
{
  unsigned int chksum=0;
  int i,j;
  
  for(i=0;i<chars;i++)
  {
    for(j=7;j>=0;j--)
    {
      if((chksum>>31) == (((unsigned int)(string[i]>>j))&1))
      {
        chksum<<=1;
      }
      else
      {
        chksum = (chksum<<1)^0x04C11DB7;
      }
    }
  }
  return chksum;
}

void checkpoint_write(unsigned int exp, int bit_min, int bit_max, unsigned int cur_class, int num_factors)
/*
checkpoint_write() writes the checkpoint file.
*/
{
  FILE *f;
  char buffer[100], filename[20];
  unsigned int i;
  
  sprintf(filename, "M%u.ckp", exp);
  
  f=fopen(filename, "w");
  if(f==NULL)
  {
    printf("WARNING, could not write checkpoint file \"%s\"\n", filename);
  }
  else
  {
    sprintf(buffer,"%u %d %d %d %s: %d %d", exp, bit_min, bit_max, NUM_CLASSES, MFAKTO_VERSION, cur_class, num_factors);
    i=checkpoint_checksum(buffer,strlen(buffer));
    fprintf(f,"%u %d %d %d %s: %d %d %08X", exp, bit_min, bit_max, NUM_CLASSES, MFAKTO_VERSION, cur_class, num_factors, i);
    fclose(f);
  }
}


int checkpoint_read(unsigned int exp, int bit_min, int bit_max, unsigned int *cur_class, int *num_factors)
/*
checkpoint_read() reads the checkpoint file and compares values for exp,
bit_min, bit_max, NUM_CLASSES read from file with current values.
If these parameters are equal than it sets cur_class and num_factors to the
values from the checkpoint file.

returns 1 on success (valid checkpoint file)
returns 0 otherwise
*/
{
  FILE *f;
  int ret=0,i,chksum;
  char buffer[100], buffer2[100], *ptr, filename[20];
  
  for(i=0;i<100;i++)buffer[i]=0;

  *cur_class=-1;
  *num_factors=0;
  
  sprintf(filename, "M%u.ckp", exp);
  
  f=fopen(filename, "r");
  if(f==NULL)
  {
    return 0;
  }
  i=fread(buffer,sizeof(char),99,f);
  sprintf(buffer2,"%u %d %d %d %s: ", exp, bit_min, bit_max, NUM_CLASSES, MFAKTO_VERSION);
  ptr=strstr(buffer, buffer2);
  if(ptr==buffer)
  {
    i=strlen(buffer2);
    if(i<70)
    {
      ptr=&(buffer[i]);
      sscanf(ptr,"%d %d", cur_class, num_factors);
      sprintf(buffer2,"%u %d %d %d %s: %d %d", exp, bit_min, bit_max, NUM_CLASSES, MFAKTO_VERSION, *cur_class, *num_factors);
      chksum=checkpoint_checksum(buffer2,strlen(buffer2));
      sprintf(buffer2,"%u %d %d %d %s: %d %d %08X", exp, bit_min, bit_max, NUM_CLASSES, MFAKTO_VERSION, *cur_class, *num_factors, chksum);
      if(*cur_class >= 0 && \
         *cur_class < NUM_CLASSES && \
         *num_factors >= 0 && \
         strlen(buffer) == strlen(buffer2) && \
         strstr(buffer, buffer2) == buffer)
      {
        ret=1;
      }
    }
  }
  fclose(f);
  return ret;
}


void checkpoint_delete(unsigned int exp)
/*
tries to delete the checkpoint file
*/
{
  char filename[20];
  sprintf(filename, "M%u.ckp", exp);
  
  if(remove(filename))
  {
    printf("WARNING: can't delete the checkpoint file \"%s\"\n", filename);
  }
}
