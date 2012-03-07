/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2011  Oliver Weihe (o.weihe@t-online.de)
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

void sieve_init();
void sieve_free();
void sieve_init_class(unsigned int exp, unsigned long long int k_start, unsigned int sieve_limit);
void sieve_candidates(int ktab_size, unsigned int *ktab, unsigned int sieve_limit);
unsigned int sieve_sieve_primes_max(unsigned int exp);
#ifdef __cplusplus
}
#endif

