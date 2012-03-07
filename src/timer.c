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
#include "timer.h"
#include "compatibility.h"

void timer_init(struct timeval *timer)
{
  gettimeofday(timer,NULL);
}


unsigned long long int timer_diff(struct timeval *timer)
/* returns the time in microseconds since "timer" was initialized with timer_init() */
{
  unsigned long long int usecs;
  struct timeval t2;
  gettimeofday(&t2,NULL);
  usecs = t2.tv_sec - timer->tv_sec;
  usecs *= 1000000;
//  usecs += t2.tv_usec - timer->tv_usec;
  usecs += t2.tv_usec;
  usecs -= timer->tv_usec;
  return usecs;
}

void timertest()
{
  struct timeval timer;
  unsigned long long int iter=0, zero=0, negative=0, positive=0;
  unsigned long long int diff, last_diff=0, min_step=1000000, max_step=0;
  
  printf("checking timer functions\n");
  
  timer_init(&timer);
  do
  {
    iter++;
    diff=timer_diff(&timer);
    if     (diff == last_diff)zero++;
    else if(diff <  last_diff)negative++;
    else
    {
      positive++;
      if((diff - last_diff) < min_step)min_step = diff - last_diff;
      if((diff - last_diff) > max_step)max_step = diff - last_diff;
    }
    last_diff=diff;
  }
  while(diff<10000000);
  printf("  %" PRIu64 " time measurements within ten seconds\n", iter);
  printf("  negative steps: %" PRIu64 "\n", negative);
  printf("  zero steps:     %" PRIu64 "\n", zero);
  printf("  positive steps: %" PRIu64 "\n", positive);
  if(negative > 0)
    printf("WARNING: %8" PRIu64 " negative timesteps\n", negative);
  printf("  smallest (non-zero) time step: %" PRIu64 "usec\n", min_step);
  printf("  biggest time step:             %" PRIu64 "usec\n", max_step);
}

void sleeptest()
{
  int i, delay, reps;
  struct timeval timer;
  unsigned long long int t, t_min, t_max, t_sum;

  for(delay = 1000; delay < 256000; delay *= 2)
  {
    t_min = 0xFFFFFFFFFFFFFFFFULL;
    t_max = 0;
    t_sum = 0;
    
    reps = 1000000 / delay;

    for(i = 0; i < reps; i++)
    {
      timer_init(&timer);
      my_usleep(delay);
      t = timer_diff(&timer);
    
      if(t < t_min)t_min = t;
      if(t > t_max)t_max = t;
      t_sum += t;
    }
    printf("my_usleep(%6u): t_min = %6" PRIu64 "us, t_max = %6" PRIu64 "us, t_avg = %6" PRIu64 "us\n", delay, t_min, t_max, t_sum / (unsigned long long int)reps);
  }
}
