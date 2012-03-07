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

#ifdef CHECKS_MODBASECASE

/*
A = limit for qi
B = step number
C = qi
D = index for modbasecase_debug[];
*/
#if defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= 200
  #define MODBASECASE_QI_ERROR(A, B, C, D) \
  if(C > (A)) \
  { \
    printf("EEEEEK, step %d qi = %u\n", B, C); \
    modbasecase_debug[D]++; \
  }
#else
  #define MODBASECASE_QI_ERROR(A, B, C, D) \
  if(C > (A)) \
  { \
    modbasecase_debug[D]++; \
  }
#endif



/*
A = q.dX
B = step number
C = number of q.dX
D = index for modbasecase_debug[];
*/
#if defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= 200
  #define MODBASECASE_NONZERO_ERROR(A, B, C, D) \
  if(A) \
  { \
    printf("EEEEEK, step %d q.d%d is nonzero: %u\n", B, C, A); \
    modbasecase_debug[D]++; \
  }
#else
  #define MODBASECASE_NONZERO_ERROR(A, B, C, D) \
  if(A) \
  { \
    modbasecase_debug[D]++; \
  }
#endif



/*
A = limit
B = step number
C = nn
D = index for modbasecase_debug[];
*/
#if defined USE_DEVICE_PRINTF && __CUDA_ARCH__ >= 200
  #define MODBASECASE_NN_BIG_ERROR(A, B, C, D) \
  if(C > A) \
  { \
    printf("EEEEEK, step %d nn.dX is too big: %u\n", B, C); \
    modbasecase_debug[D]++; \
  }
#else
  #define MODBASECASE_NN_BIG_ERROR(A, B, C, D) \
  if(C > A) \
  { \
    modbasecase_debug[D]++; \
  }
#endif

#else

#define MODBASECASE_QI_ERROR(A, B, C, D)
#define MODBASECASE_NONZERO_ERROR(A, B, C, D)
#define MODBASECASE_NN_BIG_ERROR(A, B, C, D)

#endif
