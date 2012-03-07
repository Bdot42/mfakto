/*
This file is part of mfaktc.
Copyright (C) 2009, 2010  Oliver Weihe (o.weihe@t-online.de)

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


/*
This is based on a code sniplet from Kevin (kjaget on www.mersenneforum.org)

This doesn't act like a real gettimeofday(). It has a wrong offset but this is
OK since mfaktc only uses this to measure the time difference between two calls
of gettimeofday().
*/


#include <winsock2.h>

__inline int gettimeofday(struct timeval *tv, struct timezone *tz)
{
  static LARGE_INTEGER frequency;
  static int frequency_flag = 0;

  if(!frequency_flag)
  {
    QueryPerformanceFrequency(&frequency);
    frequency_flag = 1;
  }

  if(tv)
  {
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    tv->tv_sec =  (long) (counter.QuadPart / frequency.QuadPart);
    tv->tv_usec = (long)((counter.QuadPart % frequency.QuadPart) / ((double)frequency.QuadPart / 1000000.0));
  }
  return 0;
}
