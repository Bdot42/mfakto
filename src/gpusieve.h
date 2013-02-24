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
#ifdef __cplusplus
extern "C" {
#endif

int gpusieve_init (mystuff_t *mystuff, cl_context context);
void gpusieve_init_exponent (mystuff_t *mystuff);
void gpusieve_init_class (mystuff_t *mystuff, unsigned long long k_min);
void gpusieve (mystuff_t *mystuff, unsigned long long num_k_remaining);
int gpusieve_free (mystuff_t *mystuff);

#ifdef __cplusplus
}
#endif

