/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2013  Oliver Weihe (o.weihe@t-online.de)
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

#ifndef _MSC_VER
  #include <unistd.h> // needed for usleep()
#endif

/* Windows is not POSIX-compliant */
#ifdef _MSC_VER
  #define strncasecmp _strnicmp
#endif

/* define some format strings */
#if defined __APPLE__ || _MSC_VER
  #define PRId64 "lld"
  #define PRIu64 "llu"
  #define PRIx64 "llx"
#elif __MINGW32__ || __CYGWIN__
  #include <inttypes.h>
#else
  #define PRId64 "Ld"
  #define PRIu64 "Lu"
  #define PRIx64 "Lx"
#endif


#ifdef _MSC_VER
  #define my_usleep(A) Sleep((A) / 1000)
#else
  #define my_usleep(A) usleep(A)
#endif

#ifdef __cplusplus
  #define MAX(a,b)    (((a) > (b)) ? (a) : (b))
  #define MIN(a,b)    (((a) < (b)) ? (a) : (b))
#endif
