/*
This file is part of mfakto.
Copyright (C) 2012 - 2013  Bertram Franz (bertramf@gmx.net)

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

/* This file contains functions for performance-testing of various mfakto-areas */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include "string.h"
#include "CL/cl.h"
#include "params.h"
#include "my_types.h"
#include "compatibility.h"
#include "read_config.h"
#include "parse.h"
#include "sieve.h"
#include "timer.h"
#include "checkpoint.h"
#include "filelocking.h"
#include "signal_handler.h"
#include "mfakto.h"
#ifndef _MSC_VER
#include <sys/time.h>
#else
#include "time.h"
#define time _time64
#define localtime _localtime64
#endif

extern "C" mystuff_t mystuff;
extern "C" OpenCL_deviceinfo_t deviceinfo;
extern cl_command_queue commandQueue, commandQueuePrf;

#define EXP 66362159

int init_perftest(int devicenumber)
{
  cl_uint i;
  mystuff.mode = MODE_PERFTEST;

  // always prepare 10 blocks for the perftest
  mystuff.num_streams = 10;
  init_CL(mystuff.num_streams, devicenumber);

//  i = (cl_uint)deviceinfo.maxThreadsPerBlock * deviceinfo.units * mystuff.vectorsize;
  i = 2048;
  while( (i * 2) <= mystuff.threads_per_grid_max) i = i * 2;
  mystuff.threads_per_grid = min(i, (cl_uint)deviceinfo.maxThreadsPerGrid);

  init_CLstreams();  // alloc buffers

  register_signal_handler(&mystuff);

  mystuff.sieve_primes_max_global = 1000000;
  return 0;
}

/* test the performance of the sieve init
   this performance is not essential, but for small classes (low bit-levels),
   it may be significant */
int test_sieve_init(int par)
{
  struct timeval timer;
  double time1;
  cl_uint test_sizes[] = {5000, 20000, 80000, 200000, 500000, 1000000};
  cl_uint test_loops = sizeof(test_sizes) / sizeof(test_sizes[0]);
  cl_uint i, j;
  cl_ulong k=0;

  printf("1. Sieve-Init (once per class, 960 times per test, avg. for %d iterations)\n", par);
  for (j=0; j<test_loops; j++)
  {
    timer_init(&timer);
    
    for (i=0; i<(cl_uint)par; i++)
    {
      sieve_init_class(EXP, k++, test_sizes[j]);
    }
    time1 = (double)timer_diff(&timer);
    printf("\tInit_class(sieveprimes=%7d):    %6.2f ms\n", test_sizes[j], time1/par/1000);
  }
  return 0;
}

/* Test the core sieving performance.
   This is the main part of the CPU; it's speed directly influences total
   mfakto performance. Achieving the same sieve output at a doubled SievePrimes
   value will increase total mfakto throughput by ~10% (with diminishing returns) */
int test_sieve(int par)
{
  /* This is how the resulting table looks like on a Phenom X4 955 (3.2GHz):
  2. Sieve (M/s)
SievePrimes:     256    1000    2000    3000    4000    5000   10000   20000   40000   60000   80000  100000  200000  500000 1000000
SieveSizeLimit
    12 kiB     277.7   205.6   177.3   161.2   150.3   142.3   119.5    92.8    68.8    57.0    49.5    44.4    30.3    16.0     6.1
    24 kiB     298.2   219.5   190.4   175.2   165.0   157.2   134.5   113.4    87.8    73.4    64.8    58.6    42.3    24.8    11.1
    36 kiB     301.9   223.4   192.9   177.3   167.7   159.9   139.1   118.3    97.9    83.3    73.7    66.9    49.4    30.7    15.2
    48 kiB     302.3   225.0   194.0   178.9   169.3   162.2   141.3   122.3   103.3    90.0    80.1    72.9    54.3    34.9    18.3
    59 kiB     294.8   218.8   190.4   176.3   166.0   158.9   139.6   120.8   102.9    92.6    83.2    76.1    57.0    37.5    21.4
    71 kiB     261.8   193.4   167.2   154.0   145.6   139.6   122.8   107.5    92.6    84.7    77.5    71.8    55.1    37.7    22.9
    83 kiB     243.5   177.5   153.5   141.9   133.8   128.5   113.0    98.8    85.7    78.9    73.6    68.5    53.6    37.7    24.2
    95 kiB     233.2   168.7   146.2   135.3   127.3   122.1   107.2    94.4    82.2    75.3    70.8    67.2    53.2    37.8    25.2
   118 kiB     204.2   148.2   128.6   118.5   111.9   107.4    94.8    83.9    73.8    67.7    64.3    61.8    50.2    36.9    26.3
   130 kiB     195.8   140.9   122.5   113.3   107.0   102.5    90.8    80.7    70.7    65.2    61.8    59.4    49.3    36.8    26.4
   154 kiB     182.1   131.4   113.4   105.0    99.7    95.5    84.1    75.4    66.7    61.9    58.5    56.2    48.2    35.8    26.9
   189 kiB     167.2   120.2   103.8    95.6    90.3    86.8    76.6    68.4    61.2    57.0    54.4    51.9    45.6    34.5    27.0
   224 kiB     157.2   112.6    98.1    90.0    85.1    82.3    72.5    64.6    57.8    54.2    51.6    49.3    44.1    33.9    27.0
   236 kiB     155.0   111.7    96.6    88.8    84.3    80.8    71.4    63.7    57.1    53.4    51.2    48.8    43.5    33.9    27.6
   248 kiB     152.9   109.9    95.4    88.4    83.0    80.0    70.7    63.0    56.4    53.0    50.7    48.7    43.5    34.0    27.8
   260 kiB     151.3   109.3    94.4    87.4    82.6    78.9    70.3    62.5    56.0    52.3    50.4    48.2    42.7    33.9    27.9
   295 kiB     147.0   105.7    91.9    84.7    80.3    76.9    68.1    60.8    54.5    51.1    48.8    47.3    42.3    33.4    27.6
   354 kiB     141.5   102.0    88.7    81.7    77.5    74.3    65.9    58.6    52.9    49.6    47.3    45.9    40.6    33.3    28.0
   425 kiB     136.7    99.0    85.2    79.4    75.0    72.3    63.9    56.6    51.0    48.2    45.7    44.7    39.6    33.1    27.4
   507 kiB     128.6    92.6    80.3    74.4    70.0    67.5    60.0    53.3    47.8    44.9    43.0    41.5    37.2    32.3    25.5
   590 kiB     117.2    82.8    72.3    66.1    63.2    60.1    53.8    47.8    42.8    40.1    38.9    37.2    34.2    28.8    24.2
   708 kiB     104.6    74.4    64.3    59.0    56.1    54.0    47.2    42.9    38.1    36.1    34.7    33.7    30.6    27.1    22.9
   849 kiB      96.1    67.3    57.9    54.0    50.7    49.0    43.1    38.6    35.3    32.6    31.7    30.1    27.8    24.4    22.5
  1014 kiB      88.1    62.6    53.7    49.3    47.3    45.0    39.8    35.5    32.4    30.0    29.4    28.2    25.8    23.2    19.7
  1038 kiB      87.9    61.5    53.4    49.0    45.9    45.0    39.1    35.5    32.1    30.3    28.6    28.2    26.0    22.6    19.2
  2005 kiB      74.2    50.4    44.7    41.0    37.8    38.0    32.5    29.5    26.6    25.0    23.5    24.3    21.7    17.9    17.0
Best SieveSizeLimit for
SievePrimes:     256    1000    2000    3000    4000    5000   10000   20000   40000   60000   80000  100000  200000  500000 1000000
at kiB:           48      48      48      48      48      48      48      48      48      59      59      59      59      95     354
max M/s:       302.3   225.0   194.0   178.9   169.3   162.2   141.3   122.3   103.3    92.6    83.2    76.1    57.0    37.8    28.0
Sieved out:   63.63%  69.94%  72.35%  73.58%  74.38%  74.97%  76.63%  78.08%  79.35%  80.02%  80.47%  80.81%  81.78%  82.91%  83.68%

Again, an otherwise idle Phenom X4 955 (3.2GHz):
SievePrimes:     256     396     611     945    1460    2257    3487    5389    8328   12871   19890   30738   47503   73411  113449  175323  270944  418716  647083 1000000
SieveSizeLimit
    12 kiB     261.1   237.4   217.4   202.7   190.4   173.2   156.5   140.4   125.5   111.2    93.7    77.5    63.8    51.7    41.6    32.8    25.0    18.3    12.9     6.3
    24 kiB     300.9   271.9   247.1   225.4   205.0   187.3   171.1   155.5   141.1   127.0   113.9    98.4    81.4    67.4    55.5    45.1    36.0    27.9    20.8    11.4
    36 kiB     305.2   276.4   250.8   228.2   207.6   189.4   173.6   159.3   145.8   132.2   119.1   107.5    92.0    76.8    63.5    52.6    42.7    34.0    26.1    15.6
    48 kiB     306.8   277.7   252.3   229.9   209.5   191.4   175.0   160.7   148.0   135.5   123.2   110.4    98.6    83.4    69.3    57.6    47.3    38.5    30.2    19.0
    59 kiB     300.4   272.1   247.6   225.7   206.4   188.5   173.0   159.3   146.7   135.0   123.1   111.3   100.8    87.2    73.0    61.0    50.7    41.6    32.9    22.1
    71 kiB     265.8   239.8   218.1   198.7   181.4   166.1   152.0   139.4   128.9   118.8   109.3    99.8    90.3    80.8    69.1    58.5    49.5    41.5    33.8    23.5
    83 kiB     247.3   222.7   200.8   182.5   167.1   152.9   140.1   128.2   118.5   109.4   100.1    92.1    83.5    76.3    66.4    56.9    48.7    41.1    34.0    24.8
    95 kiB     236.9   212.6   192.1   174.1   158.6   145.1   133.2   121.8   112.4   104.0    95.7    87.6    79.9    73.2    64.7    56.2    48.2    41.3    34.4    25.7
   118 kiB     208.8   186.6   168.1   152.1   139.0   127.4   117.0   107.4    99.2    91.9    85.2    78.4    72.2    65.9    60.3    52.9    46.3    40.0    33.9    26.4
   130 kiB     199.0   178.2   160.6   145.2   132.5   121.5   111.6   103.0    94.6    87.9    81.6    75.5    69.5    63.6    58.5    51.9    45.7    39.6    33.7    26.8
   154 kiB     184.8   165.8   148.9   135.3   123.3   112.6   103.7    95.7    88.4    82.0    76.4    70.7    65.4    60.1    55.8    50.2    44.3    38.9    33.0    27.4
   189 kiB     170.1   151.6   136.7   123.6   112.7   103.0    94.2    87.0    80.3    74.5    69.7    64.7    60.3    55.8    51.3    47.5    42.3    37.5    32.4    27.5
   224 kiB     159.9   142.8   128.7   116.6   106.2    97.0    89.0    82.3    75.9    70.3    65.6    61.3    57.0    52.9    48.9    45.6    41.3    36.4    31.8    28.0
   236 kiB     158.0   141.4   126.9   115.1   104.6    95.9    88.2    81.1    74.8    69.4    64.9    60.5    56.3    52.5    48.4    45.3    41.1    36.3    31.7    27.9
   248 kiB     155.8   139.2   125.0   113.7   103.5    94.6    86.9    80.0    74.1    68.6    64.2    59.6    55.7    51.8    48.0    44.9    40.7    36.2    31.7    27.3
   260 kiB     152.9   136.9   123.1   111.1   101.5    93.2    85.5    78.6    72.9    67.6    62.7    59.0    54.8    51.2    47.4    44.1    40.6    36.0    31.3    28.0
   295 kiB     148.3   132.4   119.4   108.5    98.6    90.2    83.2    76.4    70.8    65.4    61.2    57.2    53.2    49.9    46.5    43.0    39.7    35.5    31.6    26.8
   354 kiB     143.3   127.4   114.6   104.3    95.2    87.2    80.4    73.8    68.5    63.4    59.2    55.3    51.7    48.3    45.1    41.8    39.3    34.8    31.7    27.0
   425 kiB     138.3   124.1   111.2   101.3    92.1    84.9    78.0    72.0    66.6    61.7    57.5    53.5    50.5    47.3    44.1    41.1    38.1    35.0    31.0    27.3
   507 kiB     134.0   119.3   107.2    97.0    89.3    81.1    74.9    69.1    64.1    59.1    55.5    51.3    48.4    45.4    42.1    39.4    36.9    33.7    30.2    27.5
   590 kiB     122.0   108.6    97.0    88.4    80.5    73.7    67.3    62.3    57.5    53.8    49.8    46.4    43.9    40.7    38.1    36.1    33.1    31.1    27.7    26.4
   708 kiB     106.5    95.0    84.8    76.7    69.7    64.0    58.6    53.9    50.2    46.6    43.5    40.7    38.0    35.5    33.7    31.7    29.4    27.8    25.7    23.2
   849 kiB      96.7    85.6    76.3    68.7    62.5    57.0    52.7    48.3    44.9    41.7    38.9    36.5    34.1    32.1    30.4    28.1    27.0    24.8    23.8    20.7
  1014 kiB      90.3    80.2    71.3    64.8    58.5    53.3    49.0    45.1    42.3    39.2    36.5    33.8    31.9    30.2    28.3    27.4    24.8    24.0    22.2    20.1
  1038 kiB      90.1    79.5    70.6    64.2    57.8    52.7    48.6    44.7    41.5    38.8    36.2    33.7    31.5    29.7    28.5    26.5    24.7    23.4    22.6    21.7
  2005 kiB      75.0    65.4    58.4    52.5    48.2    43.4    39.9    36.7    34.3    32.1    29.2    27.8    26.4    24.9    22.5    22.7    20.7    19.5    19.7    17.2
Best SieveSizeLimit for
SievePrimes:     256     396     611     945    1460    2257    3487    5389    8328   12871   19890   30738   47503   73411  113449  175323  270944  418716  647083 1000000
at kiB:           48      48      48      48      48      48      48      48      48      48      48      59      59      59      59      59      59      59      95     260
max M/s:       306.8   277.7   252.3   229.9   209.5   191.4   175.0   160.7   148.0   135.5   123.2   111.3   100.8    87.2    73.0    61.0    50.7    41.6    34.4    28.0
Sieved out:   63.63%  65.94%  67.94%  69.73%  71.31%  72.73%  74.00%  75.16%  76.21%  77.18%  78.06%  78.88%  79.64%  80.34%  80.99%  81.60%  82.17%  82.70%  83.20%  83.68%

busy Phenom X4 955 (3.2GHz):
SievePrimes:     256     396     611     945    1460    2257    3487    5389    8328   12871   19890   30738   47503   73411  113449  175323  270944  418716  647083 1000000
SieveSizeLimit
    12 kiB     246.1   223.5   204.9   188.7   175.9   160.5   144.4   130.2   117.7   105.7    89.9    74.6    61.6    49.4    40.5    31.4    23.2    12.7     6.7     4.1
    24 kiB     253.0   233.5   214.1   198.2   174.9   169.8   153.3   144.3   132.0   119.2   107.8    94.1    78.1    64.7    54.3    43.7    33.2    18.7    11.7     7.7
    36 kiB     259.4   239.4   219.7   201.7   187.0   173.0   159.8   147.6   136.8   124.1   112.8   101.6    87.2    74.1    61.8    50.4    38.2    23.9    16.0    10.7
    48 kiB     250.8   238.7   219.6   204.9   188.0   174.1   159.2   148.9   138.3   127.3   115.8   104.8    95.0    80.8    67.3    55.5    41.2    29.7    19.7    13.7
    59 kiB     249.2   230.4   215.2   197.4   184.3   170.7   158.3   147.0   136.3   126.8   116.3   106.0    96.2    84.1    69.5    57.5    45.2    32.5    22.7    15.9
    71 kiB     228.5   208.6   191.7   179.0   164.3   150.8   139.9   129.6   120.9   110.9   103.0    95.1    86.3    77.2    65.5    54.6    43.4    33.0    24.9    17.9
    83 kiB     209.0   193.8   177.9   165.8   151.0   139.3   128.9   117.8   109.8   103.6    94.8    87.0    79.2    72.0    62.2    52.7    42.1    33.3    25.8    19.2
    95 kiB     200.2   187.5   169.6   155.7   143.9   131.9   122.4   113.6   104.9    97.4    89.4    81.1    74.7    68.2    60.5    50.8    43.1    34.6    26.8    20.4
   118 kiB     184.1   166.4   149.9   137.4   126.6   117.5   107.7   100.6    92.4    85.8    79.7    73.4    67.0    60.5    55.1    48.2    41.5    34.6    28.1    21.7
   130 kiB     176.5   159.0   144.5   132.0   121.0   111.7   103.3    95.8    88.6    82.4    76.3    70.3    64.6    58.2    54.1    47.3    41.4    34.9    28.5    22.5
   154 kiB     163.6   148.5   133.9   122.3   112.9   104.0    96.2    89.6    83.1    75.5    71.6    66.0    60.9    55.6    51.2    45.7    40.5    34.9    29.1    23.4
   189 kiB     149.7   134.8   123.2   112.1   102.8    94.9    87.2    81.2    74.5    69.7    65.0    60.6    56.1    51.6    47.6    43.5    38.4    33.9    27.9    24.2
   224 kiB     141.1   127.1   117.2   106.1    97.6    89.7    82.9    76.8    71.2    65.8    61.6    57.3    53.5    49.4    45.4    42.3    38.2    33.5    29.3    25.3
   236 kiB     139.8   125.0   114.8   104.2    96.2    88.0    81.5    76.0    69.8    65.5    61.1    56.9    52.7    49.2    44.8    42.1    38.3    33.4    29.3    25.5
   248 kiB     137.7   124.6   110.9   103.6    95.0    87.3    80.9    74.5    68.9    64.0    60.2    56.1    52.5    48.3    44.8    41.8    38.2    33.3    29.4    25.1
   260 kiB     135.0   121.6   111.1   101.6    92.6    85.7    78.8    72.8    68.2    63.6    59.1    55.2    51.2    47.6    44.3    40.5    37.3    33.3    29.1    25.6
   295 kiB     129.9   117.2   107.0    98.6    90.2    83.4    77.2    70.9    66.0    61.7    57.5    53.8    50.0    46.8    43.5    40.5    37.1    32.9    29.4    24.9
   354 kiB     125.0   112.3   102.2    94.3    86.8    80.0    74.1    68.4    64.1    59.6    55.7    51.9    48.5    45.5    42.1    39.3    37.0    32.8    29.6    25.1
   425 kiB     121.0   109.8    99.2    91.4    83.2    77.1    71.0    66.7    62.1    57.5    53.3    50.1    47.5    42.8    40.6    38.1    35.8    33.0    29.3    25.8
   507 kiB     116.7   106.2    94.8    87.9    79.7    74.1    68.9    63.4    59.0    54.6    51.9    48.2    45.3    42.7    39.9    37.4    35.1    32.1    28.7    26.0
Best SieveSizeLimit for
SievePrimes:     256     396     611     945    1460    2257    3487    5389    8328   12871   19890   30738   47503   73411  113449  175323  270944  418716  647083 1000000
at kiB:           36      36      36      48      48      48      36      48      48      48      59      59      59      59      59      59      59     154     354     507
max M/s:       259.4   239.4   219.7   204.9   188.0   174.1   159.8   148.9   138.3   127.3   116.3   106.0    96.2    84.1    69.5    57.5    45.2    34.9    29.6    26.0
Sieved out:   65.45%  67.64%  69.55%  71.24%  72.74%  74.09%  75.30%  76.40%  77.40%  78.32%  79.16%  79.94%  80.66%  81.32%  81.94%  82.52%  83.06%  83.57%  84.04%  84.50%

idle i7 2600:
SievePrimes:     256     396     611     945    1460    2257    3487    5389    8328   12871   19890   30738   47503   73411  113449  175323  270944  418716  647083 1000000
SieveSizeLimit
    12 kiB     381.5   344.8   309.7   279.2   253.5   228.4   206.1   183.5   161.7   144.1   117.6    94.3    77.3    63.0    51.0    41.0    32.4    25.3    19.2    13.6
    24 kiB     405.4   365.5   330.3   297.3   270.4   245.7   224.6   202.9   182.2   163.5   147.1   125.5   101.6    83.4    68.4    55.9    45.3    35.8    28.5    21.1
    36 kiB     407.1   366.5   331.6   298.8   272.4   248.6   227.5   207.4   187.6   171.0   152.7   138.3   117.6    96.3    79.3    65.2    53.3    43.4    34.7    26.8
    48 kiB     352.8   315.9   286.9   260.4   237.5   218.6   200.5   184.7   169.8   155.9   142.0   127.7   115.9    97.2    81.0    68.1    56.7    46.7    38.0    29.7
    59 kiB     315.0   283.4   255.7   232.7   211.9   194.7   179.8   166.1   153.4   142.1   130.5   119.0   109.3    95.9    81.2    68.7    58.0    48.6    40.1    32.3
    71 kiB     292.6   261.6   235.6   213.4   194.5   178.7   165.5   152.9   142.5   132.1   121.9   113.0   103.1    94.2    80.4    68.9    58.6    49.7    41.5    33.8
    83 kiB     281.4   251.3   225.2   204.4   186.6   171.1   157.4   146.2   135.9   125.9   117.5   109.1    99.4    92.1    80.7    69.5    59.5    50.7    42.8    35.3
    95 kiB     269.5   240.1   216.8   195.4   178.0   163.3   151.0   139.9   129.8   121.4   113.1   104.6    96.4    89.8    80.5    69.1    59.9    51.4    43.7    35.9
   118 kiB     252.0   223.3   200.1   181.2   165.8   151.6   139.8   129.4   120.5   112.7   105.3    98.3    91.5    84.8    78.4    68.5    59.7    52.0    44.6    37.6
   130 kiB     245.6   218.1   195.5   177.0   161.1   148.0   136.5   126.4   117.5   109.9   102.8    96.1    89.8    83.0    77.3    68.6    60.0    51.9    45.3    38.3
   154 kiB     237.3   210.4   188.5   170.2   155.6   142.6   131.2   121.6   113.2   105.9    99.2    93.0    87.0    80.7    75.7    68.3    60.2    53.0    46.2    39.0
   189 kiB     231.2   205.5   183.4   166.3   151.6   138.6   127.9   118.4   109.6   102.8    96.0    90.4    85.0    79.3    73.6    68.5    61.4    53.7    47.4    40.3
   224 kiB     226.9   200.0   179.5   162.4   147.7   135.5   124.8   115.3   107.3   100.0    93.8    88.1    82.5    77.5    72.4    67.9    61.4    54.7    47.6    42.0
   236 kiB     222.0   196.8   176.7   159.3   145.3   132.3   122.1   113.5   105.4    98.5    92.0    86.3    81.4    76.3    71.3    67.0    60.8    54.4    47.6    41.3
   248 kiB     218.5   193.5   172.6   156.4   142.6   130.5   119.6   111.2   103.4    96.3    90.4    85.0    79.5    75.2    70.3    65.6    60.4    53.7    47.5    41.9
   260 kiB     212.4   188.9   168.8   152.8   138.8   127.3   117.4   108.7   101.1    93.9    88.5    83.0    77.8    73.5    68.7    64.2    59.9    53.1    47.0    41.3
   295 kiB     208.4   184.3   161.7   147.9   136.1   124.4   115.0   106.3    98.7    92.4    86.2    81.3    76.6    72.2    68.0    63.7    58.8    53.2    47.5    41.5
   354 kiB     198.2   176.4   157.4   142.9   130.4   118.8   109.7   101.6    94.3    88.2    82.3    77.6    73.4    69.0    64.9    61.2    57.3    52.2    47.1    41.4
   425 kiB     190.6   169.6   151.0   137.1   124.9   114.3   105.4    97.8    91.0    84.8    79.5    74.7    70.4    66.5    62.5    59.2    55.2    51.9    46.2    41.8
   507 kiB     184.2   164.5   147.2   133.1   121.2   111.4   102.4    94.8    88.3    82.5    77.2    72.8    68.2    64.6    61.0    57.7    54.1    51.4    45.9    41.6
   590 kiB     181.0   159.9   142.7   130.2   118.5   108.5    99.9    92.7    86.5    80.5    75.2    70.8    67.1    63.3    59.8    56.7    52.8    50.8    45.5    42.2
   708 kiB     174.8   156.1   139.6   126.2   115.2   105.2    97.6    89.9    84.2    78.0    73.8    68.9    64.9    61.5    58.4    55.0    52.1    49.5    45.6    40.6
   849 kiB     171.0   152.5   136.3   123.3   112.6   103.0    95.1    87.4    82.0    76.5    72.0    67.3    64.1    60.1    57.0    53.8    51.2    47.8    44.6    42.2
  1014 kiB     167.4   149.7   133.6   120.6   110.3   100.3    92.9    85.9    80.8    75.4    70.2    66.4    62.3    58.6    56.4    52.2    50.7    47.9    44.0    43.6
  1038 kiB     168.0   148.4   133.6   121.1   110.2   100.7    92.9    86.2    79.9    75.0    69.6    65.9    62.3    58.7    56.0    53.5    49.6    48.1    44.5    42.7
  2005 kiB     160.5   143.0   126.8   114.7   104.5    96.4    87.0    81.9    76.0    70.4    66.2    61.6    58.5    55.1    52.8    50.5    47.3    43.7    43.6    39.8
Best SieveSizeLimit for
SievePrimes:     256     396     611     945    1460    2257    3487    5389    8328   12871   19890   30738   47503   73411  113449  175323  270944  418716  647083 1000000
at kiB:           36      36      36      36      36      36      36      36      36      36      36      36      36      48      59      83     189     224     224    1014
max M/s:       407.1   366.5   331.6   298.8   272.4   248.6   227.5   207.4   187.6   171.0   152.7   138.3   117.6    97.2    81.2    69.5    61.4    54.7    47.6    43.6
Sieved out:   63.63%  65.94%  67.95%  69.73%  71.31%  72.72%  74.00%  75.16%  76.22%  77.18%  78.06%  78.88%  79.64%  80.34%  80.99%  81.60%  82.17%  82.70%  83.20%  83.68%
*/

  struct timeval timer;
  double time1;
  cl_ulong k = 0;
  cl_uint i, tmp;
  printf("\n2. Sieve (M/s)\n");

#define MAX_NUM_SPS 30

  cl_uint m=13*17*19*23;
  cl_uint ssizes[MAX_NUM_SPS];  //={1,2,3,4,5,6,7,8,10,11,13,16,19,20,21,22,25,30,36,43,50,60,72,86,88,170};
  int nss=read_array(mystuff.inifile, "TestSieveSizes", MAX_NUM_SPS, ssizes);
  cl_uint sprimes[MAX_NUM_SPS];
  int nsp=read_array(mystuff.inifile, "TestSievePrimes", MAX_NUM_SPS, sprimes);
  int ii,j;

  if (nss < 1)
  {
    fprintf(stderr, "  Could not read TestSieveSizes from %s - not testing Sieve\n", mystuff.inifile);
    return -1;
  }
  if (nsp < 1)
  {
    fprintf(stderr, "  Could not read TestSievePrimes from %s - not testing Sieve\n", mystuff.inifile);
    return -1;
  }

  if (nsp>MAX_NUM_SPS) nsp=MAX_NUM_SPS;
  double peak[MAX_NUM_SPS]={0.0}, Mps;
  int peak_index[MAX_NUM_SPS]={0};
  double last_elem[MAX_NUM_SPS]={0.0};

#ifdef SIEVE_SIZE_LIMIT
  printf("Sieve size is fixed at compile time, cannot test with variable sizes. Just running 3 fixed tests.\n\n");
#endif

  printf("SievePrimes:");
  for(ii=0; ii<nsp; ii++)
  {
    printf(" %7u", sprimes[ii]);
  }
  printf("\nSieveSizeLimit");
  for (j=0;j<nss; j++)
  {
    tmp=m*ssizes[j];
    sieve_free();
#ifdef SIEVE_SIZE_LIMIT
    sieve_init();
    if (j>=3) break; // quit after 3 equal loops if we can't dynamically set the sieve size anyway
    sieve_init_class(EXP, k++, 1000000);
    printf("\n%6d kiB  ", SIEVE_SIZE/8192+1);
#else
    sieve_init(tmp, 1000000);
    sieve_init_class(EXP, k, 1000000);
    printf("\n%6d kiB  ", tmp/8192+1);
#endif

    for(ii=0; ii<nsp; ii++)
    {
      timer_init(&timer);
      for (i=0; i<(cl_uint)(par*(nsp-ii)); i++)
      {
        sieve_candidates(mystuff.threads_per_grid, mystuff.h_ktab[0], sprimes[ii]);
      }
      time1 = (double)timer_diff(&timer);
      last_elem[ii] += mystuff.h_ktab[0][mystuff.threads_per_grid-1]; // sum the last elements to get an average
      Mps =  (double)(par*(mystuff.threads_per_grid *(nsp-ii)))/time1;
      if (Mps > peak[ii])
      {
        peak[ii]=Mps;
        peak_index[ii]=j;
      }
      printf(" %7.1f", Mps);
    }
    if (mystuff.quit)
    {
      j++;
      break;
    }
  }
  printf("\nBest SieveSizeLimit for\nSievePrimes:");
  for(ii=0; ii<nsp; ii++)
  {
    printf(" %7u", sprimes[ii]);
  }

  printf("\nat kiB:     ");
  for(ii=0; ii<nsp; ii++)
  {
#ifdef SIEVE_SIZE_LIMIT
    printf(" %7u", SIEVE_SIZE/8192+1);
#else
    printf(" %7u", m*ssizes[peak_index[ii]]/8192+1);
#endif
  }
  printf("\nmax M/s:    ");
  for(ii=0; ii<nsp; ii++)
  {
    printf(" %7.1f", peak[ii]);
  }
  printf("\nSurvivors:  ");
  for(ii=0; ii<nsp; ii++)
  {
    // last_elem/nss  is the average end of the sieved block, consisting of threads_per_grid entries
    printf(" %6.2f%%", ((double)mystuff.threads_per_grid)*100.0*j/last_elem[ii]);
  }

  printf("\n\n");
  return 0;
}

/* test the performance of the memory copy to the device
   necessary for good performance, but not a lot that can be done
   about it, this is rather informational 
   - normal
   - multiple queues
   - different sizes
   - map vs. copy
   */
int test_copy(cl_uint par)
{
  struct timeval timer;
  double time1, time2;
  cl_uint i, j;
  cl_ulong k=0;
  cl_int status;
  size_t size = mystuff.threads_per_grid * sizeof(int);

  // fill some data into the arrays (not that it matter what's in there ...)
  for (i=0; i<10; i++)
  {
    sieve_candidates(mystuff.threads_per_grid, mystuff.h_ktab[i], 5000);
  }

  printf("\n3. Memory copy to GPU (blocks of %lld bytes)\n", size);

  // first, run a while to warm up the GPU (turn up clocks) without any measurement
  for (j=0; j<2; j++)
  {
    for (i=0; i<10; i++)
    {
        status = clEnqueueWriteBuffer(commandQueuePrf,
                  mystuff.d_ktab[i],
                  CL_FALSE,
                  0,
                  size,
                  mystuff.h_ktab[i],
                  0,
                  NULL,
                  NULL);

        if(status != CL_SUCCESS) 
	      {  
	          std::cout<<"Error " << status << ": Copying h_ktab(clEnqueueWriteBuffer)\n";
            return RET_ERROR;
	      }

    }
    status = clFinish(commandQueuePrf);
  }

  time1 = 0.0;

  for (j=0; j<par; j++)
  {
    timer_init(&timer);
    for (i=0; i<10; i++)
    {
        status = clEnqueueWriteBuffer(commandQueue,
                  mystuff.d_ktab[i],
                  CL_FALSE,
                  0,
                  size,
                  mystuff.h_ktab[i],
                  0,
                  NULL,
                  NULL);

        if(status != CL_SUCCESS) 
	      {  
	          std::cout<<"Error " << status << ": Copying h_ktab(clEnqueueWriteBuffer)\n";
            return RET_ERROR;
	      }

    }
    status = clFinish(commandQueue);
    time1  += (double)timer_diff(&timer);
  }
  printf("\n  Standard copy, standard queue:\n%8d MB in %6.1f ms (%6.1f MB/s) (real)\n",
      j*10*size/1024/1024, time1/1000.0, (double)(j*10*size)/time1);

  time1 = 0.0;
  time2 = 0.0;
  cl_ulong best=(cl_ulong)(-1);

  for (j=0; j<par; j++)
  {
    timer_init(&timer);
    for (i=0; i<10; i++)
    {
        status = clEnqueueWriteBuffer(commandQueuePrf,
                  mystuff.d_ktab[i],
                  CL_FALSE,
                  0,
                  size,
                  mystuff.h_ktab[i],
                  0,
                  NULL,
                  &mystuff.copy_events[i]);

        if(status != CL_SUCCESS) 
	      {  
	          std::cout<<"Error " << status << ": Copying h_ktab(clEnqueueWriteBuffer)\n";
            return RET_ERROR;
	      }

    }
    status = clFinish(commandQueuePrf);
    time1  += (double)timer_diff(&timer);
 
    cl_ulong startTime, endTime;

    for (i=0; i<10; i++)
    {
      status = clGetEventProfilingInfo(mystuff.copy_events[i],
                                CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong),
                                &startTime,
                                0);
      if(status != CL_SUCCESS)
     	{ 
	    	std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(startTime)\n";
        return RET_ERROR;
      }
      status = clGetEventProfilingInfo(mystuff.copy_events[i],
                                CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong),
                                &endTime,
                                0);
      if(status != CL_SUCCESS) 
 	    { 
		    std::cerr<< "Error " << status << " in clGetEventProfilingInfo.(endTime)\n";
        return RET_ERROR;
      }
      status = clReleaseEvent(mystuff.copy_events[i]);
      if(status != CL_SUCCESS) 
      { 
	  	  std::cerr<< "Error " << status << ": Release in event object. (clReleaseEvent)\n";
      }

//      printf("     %lld ns (%lld - %lld)\n", endTime-startTime, endTime, startTime);
      time2 += (double)(endTime-startTime);
      best = min(best, endTime-startTime);
    }
  }
  printf("\n  Standard copy, profiled queue:\n%8d MB in %6.1f ms (%6.1f MB/s) (real)\n",
      j*10*size/1024/1024, time1/1000.0, (double)(j*10*size)/time1);
  printf("%8d MB in %6.1f ms (%6.1f MB/s) (profiled data)\n",
      j*10*size/1024/1024, time2/1e6, (double)(j*10000*size)/time2);
  printf("%8d MB in %6.1f ms (%6.1f MB/s) (profiled data, peak)\n",
      size/1024/1024, (double)best/1e6, (double)(1000*size)/(double)best);

  time1 = 0.0;

  for (j=0; j<par; j++)
  {
    timer_init(&timer);
    for (i=0; i<10; i++)
    {
      if (i&1)
      {
        status = clEnqueueWriteBuffer(commandQueuePrf,
                  mystuff.d_ktab[i],
                  CL_FALSE,
                  0,
                  size,
                  mystuff.h_ktab[i],
                  0,
                  NULL,
                  NULL);
      }
      else
      {
        status = clEnqueueWriteBuffer(commandQueue,
                  mystuff.d_ktab[i],
                  CL_FALSE,
                  0,
                  size,
                  mystuff.h_ktab[i],
                  0,
                  NULL,
                  NULL);
      }

      if(status != CL_SUCCESS) 
      {  
	          std::cout<<"Error " << status << ": Copying h_ktab(clEnqueueWriteBuffer)\n";
            return RET_ERROR;
	    }

    }
    status = clFinish(commandQueuePrf);
    status = clFinish(commandQueue); // this one does not really finish the job ...
    time1  += (double)timer_diff(&timer);
  }
  printf("\n  Standard copy, two queues:\n%8d MB in %6.1f ms (%6.1f MB/s) (real)\n",
      j*10*size/1024/1024, time1/1000.0, (double)(j*10*size)/time1);


  return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

int perftest(int par, int devicenumber)
{
  struct timeval timer;
  double time1;


  init_perftest(devicenumber);

  printf("\n\nPerftest\n\n");

  if (par == 0) par=10;

  printf("Generate list of the first 10^6 primes: ");

  timer_init(&timer);
#ifdef SIEVE_SIZE_LIMIT
  sieve_init();
#else
  sieve_init(mystuff.sieve_size, mystuff.sieve_primes_max_global);
#endif
  time1 = (double)timer_diff(&timer);
  printf("%.2f ms\n\n", time1/1000.0);
  if (mystuff.quit) exit(1);

  // 1. Sieve-Init
  test_sieve_init(par);
  if (mystuff.quit) exit(1);

  // 2. Sieve
  test_sieve(par);
  if (mystuff.quit) exit(1);

  // 3. memory copy
  test_copy((cl_uint)par);
  if (mystuff.quit) exit(1);

  // 4. kernels
  printf("\n4. mfakto_cl_63 kernel\n  soon\n");

  printf("5. mfakto_cl_71 kernel\n  soon\n");

  printf("6. barrett_79 kernel\n  soon\n");

  printf("7. barrett_92 kernel\n  soon\n");

  return 0;
}

#ifdef __cplusplus
}
#endif
