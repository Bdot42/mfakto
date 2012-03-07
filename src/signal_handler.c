/*
This file is part of mfaktc.
Copyright (C) 2011  Oliver Weihe (o.weihe@t-online.de)

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

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(BUILD_OPENCL)
#include <CL/cl.h>
#endif

#include "params.h"
#include "my_types.h"
#include "compatibility.h"

mystuff_t *signal_handler_mystuff;



void my_signal_handler(int signum)
{
#ifdef _MSC_VER
/* Windows resets the signal handler to the default action once it is
invoked so we just register it again. */
  signal(signum, &my_signal_handler);
#endif
  
  if(signal_handler_mystuff->printmode == 1)printf("\n");
  signal_handler_mystuff->quit++;
  if(signal_handler_mystuff->quit == 1)
  {
    printf("mfaktc will exit once the current class is finished.\n");
    printf("press ^C again to exit imediately\n");
  }
  if(signal_handler_mystuff->quit > 1)
  {
    printf("mfaktc will exit NOW!\n");
    exit(1);
  }
  signum++; /* useless but avoids warning about unused variable... */
}


void register_signal_handler(mystuff_t *mystuff)
{
  signal_handler_mystuff = mystuff;
  signal(SIGINT, &my_signal_handler);
}
