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
#include <stdlib.h>
#include <string.h>

#include "params.h"
#include "compatibility.h"

/* yeah, I like global variables :) */
static unsigned int *sieve,*sieve_base,mask0[32],mask1[32];
static int primes[SIEVE_PRIMES_MAX],k_init[SIEVE_PRIMES_MAX],last_sieve;


/* the sieve_table contains the number of bits set in n (sieve_table[n][8]) and
the position of the set bits
(sieve_table[n][0]...sieve_table[n][<number of bits set in n>-1])
for 0 <= n < 256 */
static unsigned int sieve_table[256][9];

static __inline unsigned int sieve_get_bit(unsigned int *array,unsigned int bit)
{
  unsigned int chunk;
  chunk=bit>>5;
  bit&=0x1F;
  return array[chunk]&mask1[bit];
}

static __inline void sieve_set_bit(unsigned int *array,unsigned int bit)
{
  unsigned int chunk;
  chunk=bit>>5;
  bit&=0x1F;
  array[chunk]|=mask1[bit];
}

static __inline void sieve_clear_bit(unsigned int *array,unsigned int bit)
{
  unsigned int chunk;
  chunk=bit>>5;
  bit&=0x1F;
  array[chunk]&=mask0[bit];
}
//#define sieve_clear_bit(ARRAY,BIT) asm("btrl  %0, %1" : /* no output */ : "r" (BIT), "m" (*ARRAY) : "memory", "cc" )
//#define sieve_clear_bit(ARRAY,BIT) ARRAY[BIT>>5]&=mask0[BIT&0x1F]

unsigned int* sieve_malloc(unsigned int size)
{
  unsigned int *array;
  if(size==0)return NULL;
  array=(unsigned int*)malloc((((size-1)>>5)+1)*4);
  return array;
}

void sieve_init()
{
  int i,j,k;
  for(i=0;i<32;i++)
  {
    mask1[i]=1<<i;
    mask0[i]=0xFFFFFFFF-mask1[i];
  }
  sieve=sieve_malloc(SIEVE_SIZE);
  sieve_base=sieve_malloc(SIEVE_SIZE);

  primes[0]=3;
  i=0;j=3;
  while(i<SIEVE_PRIMES_MAX)
  {
    k=0;
    while(primes[k]*primes[k]<=j)
    {
      if(j%primes[k]==0)
      {
        j+=2;
        k=0;
      }
      else
      {
        k++;
      }
    }
    primes[i]=j;
    i++;    
    j+=2;
  }

  for(i=0;i<256;i++)
  {
    sieve_table[i][8]=0;
    for(j=0;j<8;j++)
    {
      if(i&(1<<j))
      {
        sieve_table[i][sieve_table[i][8]++]=j;
      }
    }
  }
}

void sieve_free()
{
  free(sieve);
  free(sieve_base);
}

int sieve_euclid_modified(int j, int n, int r)
/*
(k*j) % n = r
calculates and returns k 
*/
{
/* using long long int because I'm too lazy to figure out where I can live with
smaller ints */
  long long int nn,nn_old,jj,pi0,pi1,pi2,qi0,qi1,qi2,tmp;
  
  nn_old=n;
  jj=j;
  
  if(r==0)return 0;	/* trivially! */
  if(j==1)return r;	/* easy, isn't it? */

/*** step 0 ***/
  qi0=nn_old/jj;
  nn=nn_old%jj;
  pi0=0;
  nn_old=jj;
  jj=nn;

  if(jj==1)		/* qi0 * j = -1 mod n     FIXME, is the description correct?*/
  {
    tmp=((-r*qi0) % n) + n;
    return (int)tmp;
  }

/*** step 1 ***/
  qi1=qi0;
  qi0=nn_old/jj;
  nn=nn_old%jj;
  pi1=pi0;
  pi0=r;
  nn_old=jj;
  jj=nn;

/*** step 2+ ***/
  while(nn)
  {
    qi2=qi1;
    qi1=qi0;
    qi0=nn_old/jj;
    nn=nn_old%jj;
    pi2=pi1;
    pi1=pi0;
    pi0=(pi2-pi1*qi2)%n;
    if(pi0<0)pi0+=n;
    nn_old=jj;
    jj=nn;
  }

  tmp=(pi1-pi0*qi1)%n;
  if(tmp<0)tmp+=n;

  return (int)tmp;
}

void sieve_init_class(unsigned int exp, unsigned long long int k_start, int sieve_limit)
{
  int i,j,k,p;
  int ii,jj;

#ifdef MORE_CLASSES  
  for(i=4;i<sieve_limit;i++)
#else
  for(i=3;i<sieve_limit;i++)
#endif  
  {
    unsigned long long int check;

    p=primes[i];  
    k=0;
// oldest version, explains what happens here a little bit */    
//    while((2 * (exp%p) * ((k_start+k*NUM_CLASSES)%p)) %p != (p-1))k++;


/* second version, expensive mod is avoided as much as possible, but it is
still a brute force trail&error method */
/*    ii=(2 * (exp%p) * (k_start%p))%p;
    jj=(2 * (exp%p) * (NUM_CLASSES%p))%p;
    while(ii != (p-1))
    {
      ii+=jj;
      if(ii>=p)ii-=p;
      k++;
    }
    k_init[i]=k;*/

/* third version using a modified euclidean algorithm */
    // ii=(2ULL * (exp%p) * (k_start%p))%p;
    // jj=(2ULL * (exp%p) * (NUM_CLASSES%p))%p;     // PERF: skip %p for NUM_CLASSES

    // skip 3 modulo's and the error checking: saves 10-20 CPU-ms per class
    ii = (2ULL * (unsigned long long int)exp * (k_start%p))%p;
    jj = (9240ULL * (unsigned long long int)exp)%p;

    k = sieve_euclid_modified(jj, p, p-(1+ii));
    k_init[i]=k;

// error checking
/*    check = k_start + (unsigned long long int) k * NUM_CLASSES;
    check %= p;
    check *= exp;
    check <<= 1;
    check %= p;
    if(k < 0 || k >= p || check != (p-1))
    {
      printf("calculation of k_init[%d] failed!\n",i);
      printf("  k_start= %" PRIu64 "\n",k_start);
      printf("  exp= %u\n",exp);
      printf("  ii= %d\n",ii);
      printf("  jj= %d\n",jj);
      printf("  k= %d\n",k);
      printf("  p= %d\n",p);
      printf("  check= %" PRId64 "\n",check);
    } */
  }
  
  for(i=0;i<SIEVE_SIZE;i++)sieve_set_bit(sieve_base,i);

#ifdef MORE_CLASSES
/* presieve 13, 17, 19 and 23 in sieve_base */
  for(i=4;i<=7;i++)
#else  
/* presieve 11, 13, 17 and 19 in sieve_base */
  for(i=3;i<=6;i++)
#endif
  {
    j=k_init[i];
    p=primes[i];
    while(j<SIEVE_SIZE)
    {
//if((2 * (exp%p) * ((k_start+j*NUM_CLASSES)%p)) %p != (p-1))printf("EEEK: sieve: p=%d j=%d k=%" PRIu64 "\n",p,j,k_start+j*NUM_CLASSES);
      sieve_clear_bit(sieve_base,j);
      j+=p;
    }
//    k_init[i]=j-SIEVE_SIZE;
  }
  last_sieve = SIEVE_SIZE;
}


void sieve_candidates(int ktab_size, unsigned int *ktab, int sieve_limit)
{
  int i=-1,ii,j,k=0,p,c=0,ic;
  unsigned int s,sieve_table_8,*sieve_table_;
  unsigned int mask; //, index, index_max;
  unsigned int *ptr, *ptr_max;

#ifdef RAW_GPU_BENCH
//  quick hack to "speed up the siever", used for GPU-code benchmarks  
  for(i=0;i<ktab_size;i++)ktab[i]=i;
  return;
#endif  

  if(last_sieve < SIEVE_SIZE)
  {
    i=last_sieve;
    c=-i;
    goto _ugly_goto_in_siever;
  }

  while(k<ktab_size)
  {
//printf("sieve_candidates(): main loop start\n");
    memcpy(sieve, sieve_base, ((SIEVE_SIZE-1)>>3)+1);

/*
The first few primes in the sieve have their own code. Since they are small
they have many iterations in the inner loop. At the cost of some
initialisation we can avoid calls to sieve_clear_bit() which calculates
chunk and bit position in chunk on each call.
Every 32 iterations they hit the same bit position so we can make use of
this behaviour and precompute them. :)
*/
#ifdef MORE_CLASSES
    for(i=7;i<SIEVE_SPLIT;i++)
#else
    for(i=6;i<SIEVE_SPLIT;i++)
#endif
    {
      j=k_init[i];
      p=primes[i];
//printf("sieve: %d\n",p);
      for(ii=0; ii<32; ii++)
      {
        mask = mask0[j & 0x1F];
/*
        index = j >> 5;
        index_max = SIEVE_SIZE >> 5;
        if(index_max + (j & 0x1F) < SIEVE_SIZE)index_max++;
        while(index < index_max)
        {
          sieve[index] &= mask;
          index += p;
        }
        j+=p;
      }
      j = (index<<5) + ((j-p) & 0x1F);
      while(j>=SIEVE_SIZE)j-=p;
      j+=p;
      k_init[i]=j-SIEVE_SIZE;*/

        ptr = &(sieve[j>>5]);
        ptr_max = &(sieve[SIEVE_SIZE >> 5]);
        if( (j & 0x1F) < (SIEVE_SIZE & 0x1F))ptr_max++;
        while(ptr < ptr_max) /* inner loop, lets kick out some bits! */
        {
          *ptr &= mask;
          ptr += p;
        }
        j+=p;
      }
      j = ((int)(ptr - sieve)<<5) + ((j-p) & 0x1F); /* D'oh! Pointer arithmetic... but it is faster! */
      j -= SIEVE_SIZE;
      k_init[i] = j % p;
    }

    for(i=SIEVE_SPLIT;i<sieve_limit;i++)
    {
      j=k_init[i];
      p=primes[i];
//printf("sieve: %d\n",p);
      while(j<SIEVE_SIZE)
      {
        sieve_clear_bit(sieve,j);
        j+=p;
      }
      k_init[i]=j-SIEVE_SIZE;
    }
    
/*
we have finished sieving and now we need to translate the remaining bits in
the sieve to the correspondic k_tab offsets
*/    

/* part one of the loop:
Get the bits out of the sieve until i is a multiple of 32
this is going to fail if ktab has less than 32 elements! */
    for(i=0;(i<SIEVE_SIZE) && (i&0x1F);i++)
    {
_ugly_goto_in_siever:
      if(sieve_get_bit(sieve,i))
      {
        ktab[k++]=i+c;
        if(k >= ktab_size)
        {
          last_sieve=i+1;
          return;
        }
      }
    }
/* part two of the loop:
Get the bits out of the sieve until
a) we're close the end of the sieve
or
b) ktab is nearly filled up */
    for(;i<(SIEVE_SIZE&0xFFFFFFE0) && k<(ktab_size-33);i+=32)	// thirty-three!!!
    {
      ic=i+c;
      s=sieve[i>>5];
//#define SIEVER_OLD_METHOD
#ifdef SIEVER_OLD_METHOD
      sieve_table_=sieve_table[ s     &0xFF];
      for(p=0;p<sieve_table_[8];p++) ktab[k++]=ic   +sieve_table_[p];
      
      sieve_table_=sieve_table[(s>>8 )&0xFF];
      for(p=0;p<sieve_table_[8];p++) ktab[k++]=ic +8+sieve_table_[p];
      
      sieve_table_=sieve_table[(s>>16)&0xFF];
      for(p=0;p<sieve_table_[8];p++) ktab[k++]=ic+16+sieve_table_[p];
      
      sieve_table_=sieve_table[ s>>24      ];
      for(p=0;p<sieve_table_[8];p++) ktab[k++]=ic+24+sieve_table_[p];

#else // not SIEVER_OLD_METHOD
      sieve_table_=sieve_table[ s     &0xFF];
      sieve_table_8=sieve_table_[8];
      ktab[k  ]=ic+sieve_table_[0];
      ktab[k+1]=ic+sieve_table_[1];
      ktab[k+2]=ic+sieve_table_[2];
      ktab[k+3]=ic+sieve_table_[3];
      if(sieve_table_8>4)
      {
        ktab[k+4]=ic+sieve_table_[4];
        ktab[k+5]=ic+sieve_table_[5];
        ktab[k+6]=ic+sieve_table_[6];
        ktab[k+7]=ic+sieve_table_[7];
      }
      k+=sieve_table_8;
      
      sieve_table_=sieve_table[(s>>8 )&0xFF];
      sieve_table_8=sieve_table_[8];
      ic+=8;
      ktab[k  ]=ic+sieve_table_[0];
      ktab[k+1]=ic+sieve_table_[1];
      ktab[k+2]=ic+sieve_table_[2];
      ktab[k+3]=ic+sieve_table_[3];
      if(sieve_table_8>4)
      {
        ktab[k+4]=ic+sieve_table_[4];
        ktab[k+5]=ic+sieve_table_[5];
        ktab[k+6]=ic+sieve_table_[6];
        ktab[k+7]=ic+sieve_table_[7];
      }
      k+=sieve_table_8;
      
      sieve_table_=sieve_table[(s>>16)&0xFF];
      sieve_table_8=sieve_table_[8];
      ic+=8;
      ktab[k  ]=ic+sieve_table_[0];
      ktab[k+1]=ic+sieve_table_[1];
      ktab[k+2]=ic+sieve_table_[2];
      ktab[k+3]=ic+sieve_table_[3];
      if(sieve_table_8>4)
      {
        ktab[k+4]=ic+sieve_table_[4];
        ktab[k+5]=ic+sieve_table_[5];
        ktab[k+6]=ic+sieve_table_[6];
        ktab[k+7]=ic+sieve_table_[7];
      }
      k+=sieve_table_8;
      
      sieve_table_=sieve_table[ s>>24      ];
      sieve_table_8=sieve_table_[8];
      ic+=8;
      ktab[k  ]=ic+sieve_table_[0];
      ktab[k+1]=ic+sieve_table_[1];
      ktab[k+2]=ic+sieve_table_[2];
      ktab[k+3]=ic+sieve_table_[3];
      if(sieve_table_8>4)
      {
        ktab[k+4]=ic+sieve_table_[4];
        ktab[k+5]=ic+sieve_table_[5];
        ktab[k+6]=ic+sieve_table_[6];
        ktab[k+7]=ic+sieve_table_[7];
      }
      k+=sieve_table_8;
#endif      
    }
/* part three of the loop:
Get the bits out of the sieve until
a) sieve ends
or
b) ktab is full */    
    for(;i<SIEVE_SIZE;i++)
    {
      if(sieve_get_bit(sieve,i))
      {
        ktab[k++]=i+c;
        if(k >= ktab_size)
        {
          last_sieve=i+1;
          return;
        }
      }
    }
    c+=SIEVE_SIZE;
  }
  last_sieve=i;
}


unsigned int sieve_sieve_primes_max(unsigned int exp)
/* returns min(SIEVE_PRIMES_MAX, number of primes below exp) */
{
  int ret = 0;
  if(exp > (unsigned int)primes[SIEVE_PRIMES_MAX - 1]) ret = SIEVE_PRIMES_MAX;
  else while((unsigned int)primes[ret] < exp)ret++;
  
  return ret;
}
