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

/* main performance test function
   input: guidline for the result's precision - used for deriving the number of test repetitions
          minimum: 1, no maximum, < 1 sets default of 10
          Higher takes longer, but yields more accurate results.
          */

#ifdef __cplusplus
extern "C" {
#endif

int perftest(int par, int devicenumber);

#ifdef __cplusplus
}
#endif
