/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2014  Oliver Weihe (o.weihe@t-online.de)
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

void checkpoint_write(unsigned int exp, int bit_min, int bit_max, unsigned int cur_class, int num_factors, char* factors_string, unsigned long long int bit_level_time);
int checkpoint_read(unsigned int exp, int bit_min, int bit_max, unsigned int* cur_class, int *num_factors, char* factors_string, unsigned long long int* bit_level_time, int verbosity);
void checkpoint_delete(unsigned int exp);
