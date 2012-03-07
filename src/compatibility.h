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

#ifndef _MSC_VER
  #include <unistd.h> // needed for usleep()
#endif


/* define some format strings */
#ifdef _MSC_VER
  #define PRId64 "lld"
  #define PRIu64 "llu"
  #define PRIx64 "llx"
  
  #define strncasecmp _strnicmp
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
