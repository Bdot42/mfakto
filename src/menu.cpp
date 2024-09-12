/*
This file is part of mfaktc (mfakto).
Copyright (C) 2014  Oliver Weihe (o.weihe@t-online.de)
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

/* code to handle dynamic changing of settings */

#if defined(_MSC_VER) || defined(__MINGW32__)
  #include <conio.h>
#else
  #include "kbhit.h"
  static keyboard kb;
  #define _kbhit kb.kbhit
  #define _getch kb.getch
#endif

#include "mfakto.h"
#include "compatibility.h"
#include "sieve.h"
#include "gpusieve.h"
#include <stdio.h>
#include <iostream>


static void print_menu(mystuff_t *mystuff)
{
  puts("\nSettings menu\n\nNum  Setting     Current value  (shortcut outside of the menu for de-/increasing this setting)\n");

  printf("  1  SievePrimes       = %-8u  (-/+)\n", mystuff->sieve_primes);
  printf("  2  SieveSize         = %-8u  (s/S)\n", mystuff->gpu_sieving ? mystuff->gpu_sieve_size/1024/1024 : mystuff->sieve_size/8192);
  printf("  3  SieveProcessSize  = %-8u  (p/P)\n", mystuff->gpu_sieving ? mystuff->gpu_sieve_processing_size/1024 : mystuff->sieve_size/8192);
  printf("  4  SievePrimesAdjust = %-8u  (a/A)\n", mystuff->gpu_sieving ? 0 : mystuff->sieve_primes_adjust);
  printf("  5  FlushInterval     = %-8u  (f/F)\n", mystuff->flush);
  printf("  6  Verbosity         = %-8u  (v/V)\n", mystuff->verbosity);
  printf("  7  PrintMode         = %-8u  (r/R)\n", mystuff->printmode);
  printf("  8  Kernel            = %s  (k/K)\n", mystuff->stats.kernelname);

  printf("\n  0  Done (continue factoring)\n");
  printf("\n -1  Exit mfakto (q/Q)\n");
}

static void validate_settings(mystuff_t *mystuff)
{
  if (mystuff->gpu_sieving)
  {
    if (mystuff->sieve_primes < GPU_SIEVE_PRIMES_MIN) mystuff->sieve_primes = GPU_SIEVE_PRIMES_MIN;
    if (mystuff->sieve_primes > GPU_SIEVE_PRIMES_MAX) mystuff->sieve_primes = GPU_SIEVE_PRIMES_MAX;

    mystuff->gpu_sieve_processing_size = ((mystuff->gpu_sieve_processing_size + 4096) / 8192) * 8192;
    if (mystuff->gpu_sieve_processing_size < GPU_SIEVE_PROCESS_SIZE_MIN*1024) mystuff->gpu_sieve_processing_size = GPU_SIEVE_PROCESS_SIZE_MIN*1024;
    if (mystuff->gpu_sieve_processing_size > GPU_SIEVE_PROCESS_SIZE_MAX*1024) mystuff->gpu_sieve_processing_size = GPU_SIEVE_PROCESS_SIZE_MAX*1024;

    mystuff->gpu_sieve_size = ((mystuff->gpu_sieve_size + 512*1024) / 1024 / 1024) * 1024*1024;
    if (mystuff->gpu_sieve_size < GPU_SIEVE_SIZE_MIN * 1024*1024) mystuff->gpu_sieve_size = GPU_SIEVE_SIZE_MIN * 1024*1024;
    if (mystuff->gpu_sieve_size > GPU_SIEVE_SIZE_MAX * 1024*1024) mystuff->gpu_sieve_size = GPU_SIEVE_SIZE_MAX * 1024*1024;
    while (mystuff->gpu_sieve_size % mystuff->gpu_sieve_processing_size != 0)
      mystuff->gpu_sieve_size -= 1024*1024;   // sieve_size must be a multiple of sieve_processing_size
    while (mystuff->gpu_sieve_size < GPU_SIEVE_SIZE_MIN * 1024*1024)
      mystuff->gpu_sieve_size += 3*1024*1024;  // make sure it's not too low
    // if (mystuff->flush < 0) ... // it's an unsigned int -> huge ints will cause the class to be flushed only implicitly after the last block
  }
  else
  {
    if (mystuff->sieve_primes < SIEVE_PRIMES_MIN) mystuff->sieve_primes = SIEVE_PRIMES_MIN;
    if (mystuff->sieve_primes > SIEVE_PRIMES_MAX) mystuff->sieve_primes = SIEVE_PRIMES_MAX;
#ifndef SIEVE_SIZE_LIMIT
    if (mystuff->sieve_size < 13*17*19*23) mystuff->sieve_size = 13*17*19*23;
    else mystuff->sieve_size = mystuff->sieve_size / (13*17*19*23) * (13*17*19*23);
#endif
    if (mystuff->sieve_primes_adjust > 1) mystuff->sieve_primes_adjust = 1;
  }
  if (mystuff->verbosity < 0) mystuff->verbosity = 0;
  if (mystuff->printmode > 1) mystuff->printmode = 1;
}

static void set_sieve_primes(mystuff_t *mystuff, int new_value)
{
  mystuff->sieve_primes = new_value;
  validate_settings(mystuff);
}

static void lower_sieve_primes(mystuff_t *mystuff)
{
  mystuff->sieve_primes = mystuff->sieve_primes * 7 / 8;
  validate_settings(mystuff);
}

static void increase_sieve_primes(mystuff_t *mystuff)
{
  mystuff->sieve_primes = mystuff->sieve_primes * 9 / 8;
  validate_settings(mystuff);
}

static void set_sieve_size(mystuff_t *mystuff, int new_value)
{
  if (mystuff->gpu_sieving)
  {
    mystuff->gpu_sieve_size = new_value * 1024 * 1024;
  }
#ifndef SIEVE_SIZE_LIMIT
  else
  {
    mystuff->sieve_size = new_value * 8192 / (13*17*19*23) * (13*17*19*23);
  }
#endif
  validate_settings(mystuff);
}

static void lower_sieve_size(mystuff_t *mystuff)
{
  if (mystuff->gpu_sieving)
  {
    mystuff->gpu_sieve_size = mystuff->gpu_sieve_size * 7 / 8;
  }
#ifndef SIEVE_SIZE_LIMIT
  else
  {
    mystuff->sieve_size -=  13*17*19*23;
  }
#endif
  validate_settings(mystuff);
}

static void increase_sieve_size(mystuff_t *mystuff)
{
  if (mystuff->gpu_sieving)
  {
    mystuff->gpu_sieve_size = mystuff->gpu_sieve_size * 9 / 8;
  }
#ifndef SIEVE_SIZE_LIMIT
  else
  {
    mystuff->sieve_size +=  13*17*19*23;
  }
#endif
  validate_settings(mystuff);
}

static void set_sieve_process_size(mystuff_t *mystuff, int new_value)
{
  if (mystuff->gpu_sieving)
  {
    mystuff->gpu_sieve_processing_size = new_value * 1024;
    validate_settings(mystuff);
  }
}

static void lower_sieve_process_size(mystuff_t *mystuff)
{
  if (mystuff->gpu_sieving)
  {
    mystuff->gpu_sieve_processing_size -= 8192;
    validate_settings(mystuff);
  }
}

static void increase_sieve_process_size(mystuff_t *mystuff)
{
  if (mystuff->gpu_sieving)
  {
    mystuff->gpu_sieve_processing_size += 8192;
    validate_settings(mystuff);
  }
}

static void set_kernel(mystuff_t *mystuff, const char * new_kernel)
{
  printf("set kernel: not yet implemented\n");
}

static void use_previous_kernel(mystuff_t *mystuff)
{
  printf("prev kernel: not yet implemented\n");
}

static void use_next_kernel(mystuff_t *mystuff)
{
  printf("next kernel: not yet implemented\n");
}

static void handle_menu(mystuff_t *mystuff)
{
  int choice, new_value;
  char choice_string[10], new_value_string[50];

  for(;;)
  {
    print_menu(mystuff);
    std::cout << "Change setting number: ";
    fgets(choice_string, 9, stdin); // std:cin does not allow empty input
    choice = atoi(choice_string);
    if (choice == -1)       // quit
    {
      if (mystuff->verbosity > 0) printf("\nExiting\n");
      mystuff->quit++;
      break;
    }
    if (choice == 0)
    {
      if (mystuff->verbosity > 0) printf("\nContinuing\n");
      break; // continue
    }
    std::cout << "Change setting number " << choice << " to:";
    std::cin >> new_value_string;
    new_value = atoi(new_value_string);
    switch (choice)
    {
      case 1: set_sieve_primes(mystuff, new_value);
              break;
      case 2: set_sieve_size(mystuff, new_value);
              break;
      case 3: set_sieve_process_size(mystuff, new_value);
              break;
      case 4: mystuff->sieve_primes_adjust = new_value;
              break;
      case 5: mystuff->flush = new_value;
              break;
      case 6: mystuff->verbosity = new_value;
              break;
      case 7: mystuff->printmode = new_value;
              break;
      case 8: set_kernel(mystuff, new_value_string);
              break;
      default: std::cout << new_value_string << "? What's this?\n";
              break;
    }
  }
}

void check_and_do_reinit(mystuff_t *saved_mystuff, mystuff_t *mystuff)
{
  if (mystuff->gpu_sieving)
  {
    if (mystuff->sieve_primes != saved_mystuff->sieve_primes ||
        mystuff->gpu_sieve_processing_size != saved_mystuff->gpu_sieve_processing_size ||
        mystuff->gpu_sieve_size != saved_mystuff->gpu_sieve_size)
    {
      if (mystuff->verbosity > 0)
        printf("Reinitializing GPU sieve\n");
      gpusieve_free(mystuff);
      init_CLstreams(1);
      gpusieve_init_exponent(mystuff);
    }
  }
  else
  {
    if (mystuff->sieve_primes > saved_mystuff->sieve_primes //smaller is no problem here
#ifndef SIEVE_SIZE_LIMIT
      || mystuff->sieve_size != saved_mystuff->sieve_size
#endif
      )
    {
      if (mystuff->verbosity > 0)
        printf("Reinitializing CPU sieve\n");
      sieve_free();
#ifdef SIEVE_SIZE_LIMIT
      sieve_init();
#else
      sieve_init(mystuff->sieve_size, mystuff->sieve_primes_max);
#endif
    }
  }
}

int handle_kb_input(mystuff_t *mystuff)
{
  int last_key = 0;
  mystuff_t saved_mystuff = *mystuff;

  while (_kbhit())
  {
    switch (last_key = _getch())
    {
    case 'm': handle_menu(mystuff);
      break;
    case '-': lower_sieve_primes(mystuff);
              if (mystuff->verbosity > 0)
                printf("\nDecrease SievePrimes to %u\n", mystuff->sieve_primes);
      break;
    case '+': increase_sieve_primes(mystuff);
              if (mystuff->verbosity > 0)
                printf("\nIncrease SievePrimes to %u\n", mystuff->sieve_primes);
      break;
    case 's': lower_sieve_size(mystuff);
              if (mystuff->verbosity > 0)
                printf("\nDecrease %sSieveSize to %u\n", mystuff->gpu_sieving?"GPU":"",
                   mystuff->gpu_sieving?mystuff->gpu_sieve_size/1048576:mystuff->sieve_size);
      break;
    case 'S': increase_sieve_size(mystuff);
              if (mystuff->verbosity > 0)
                printf("\nIncrease %sSieveSize to %u\n", mystuff->gpu_sieving?"GPU":"",
                   mystuff->gpu_sieving?mystuff->gpu_sieve_size/1048576:mystuff->sieve_size);
      break;
    case 'p': lower_sieve_process_size(mystuff);
              if ((mystuff->verbosity > 0) && mystuff->gpu_sieving)
                printf("\nDecrease GPUSieveProcessSize to %u\n", mystuff->gpu_sieve_processing_size/1024);
      break;
    case 'P': increase_sieve_process_size(mystuff);
              if ((mystuff->verbosity > 0) && mystuff->gpu_sieving)
                printf("\nIncrease GPUSieveProcessSize to %u\n", mystuff->gpu_sieve_processing_size/1024);
      break;
    case 'a': if (mystuff->verbosity > 0)
                printf("\nSetting SievePrimesAdjust to 0 (was %u)\n", mystuff->sieve_primes_adjust);
              mystuff->sieve_primes_adjust=0;
      break;
    case 'A': if (mystuff->verbosity > 0)
                printf("\nSetting SievePrimesAdjust to 1 (was %u)\n", mystuff->sieve_primes_adjust);
              mystuff->sieve_primes_adjust=1;
      break;
    case 'f': if (mystuff->flush > 0)
              {
                if (mystuff->verbosity > 0)
                  printf("\nSetting FlushInterval to %u (was %u)\n", mystuff->flush - 1, mystuff->flush);
                mystuff->flush--;
              }
      break;
    case 'F': if (mystuff->verbosity > 0)
                printf("\nSetting FlushInterval to %u (was %u)\n", mystuff->flush + 1, mystuff->flush);
              mystuff->flush++;
      break;
    case 'v': if (mystuff->verbosity > 0)
              {
                mystuff->verbosity--;
                printf("\nDecrease Verbosity to %u\n", mystuff->verbosity);
              }
      break;
    case 'V': mystuff->verbosity++;
              if (mystuff->verbosity > 0) printf("\nIncrease Verbosity to %u\n", mystuff->verbosity);
      break;
    case 'r': if (mystuff->verbosity > 0)
                printf("\nSetting PrintMode to 0 (was %u)\n", mystuff->printmode);
              mystuff->printmode=0;
      break;
    case 'R': if (mystuff->verbosity > 0)
                printf("\nSetting PrintMode to 1 (was %u)\n", mystuff->printmode);
              mystuff->printmode=1;
      break;
    case 'k': use_previous_kernel(mystuff);
      break;
    case 'K': use_next_kernel(mystuff);
      break;
    case 'q':
    case 'Q': if (mystuff->verbosity > 0)
                printf("\n%c: Exiting\n", last_key);
              mystuff->quit++;
      break;
    default: printf("\nIgnoring unknown keyboard input '%c'\n", last_key);
      break;
    }
  }
  check_and_do_reinit(&saved_mystuff, mystuff);

  return last_key;
}
