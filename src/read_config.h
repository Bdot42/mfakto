/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009, 2013  Oliver Weihe (o.weihe@t-online.de)

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

#include "my_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int read_config(mystuff_t *mystuff);
int read_array(char * filename, char *name, cl_uint num, cl_uint *arr);
int my_read_int(char* inifile, char* name, int* value);
int my_read_string(char* inifile, char* name, char* string, unsigned int len);

#ifdef __cplusplus
}
#endif
