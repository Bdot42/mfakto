/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2012  Oliver Weihe (o.weihe@t-online.de)
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

#include <fcntl.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <time.h>

#ifdef _MSC_VER
  #include <Windows.h>
  #include <io.h>
  #undef open
  #undef close
  #define open _open
  #define close _close
  #define MODE _S_IREAD | _S_IWRITE
#else
  #include <unistd.h>
  #include <sched.h>
  #define MODE S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH
  static void Sleep(unsigned int ms)
  {
    struct timespec ts;
    ts.tv_sec  = (time_t) ms/1000;
    ts.tv_nsec = (ms % 1000) * 1000000;
    nanosleep(&ts, NULL);
  }
#endif

#define MAX_LOCKED_FILES 5

typedef struct _lockinfo
{
  int       lockfd;
  FILE *    open_file;
  char      lock_filename[256];
} lockinfo;

static unsigned int num_locked_files = 0;
static lockinfo     locked_files[MAX_LOCKED_FILES];

/* See if the given file exists */

int file_exists (char	*filename)
{
	int fd = open(filename, _O_RDONLY | _O_BINARY);
	if (fd < 0) return 0;
	close(fd);
	return 1;
}

FILE *fopen_and_lock(const char *path, const char *mode)
{
  unsigned int i;
  int lockfd;
  FILE *f;

  if (strlen(path) > 250)
  {
    fprintf(stderr, "Cannot open %.250s: Name too long.\n", path);
    return NULL;
  }

  if (num_locked_files >= MAX_LOCKED_FILES)
  {
    fprintf(stderr, "Cannot open %.250s: Too many locked files.\n", path);
    return NULL;
  }

  sprintf(locked_files[num_locked_files].lock_filename, "%.250s.lck", path);

  for(i=0;;)
  {
    if ((lockfd = open(locked_files[num_locked_files].lock_filename, O_EXCL | O_CREAT, MODE)) < 0)
    {
      if (errno == EEXIST)
      {
        if (i==0) fprintf(stderr, "%.250s already exists, waiting ...\n", locked_files[num_locked_files].lock_filename);
        if (i<1000) i++; // slowly increase sleep time up to 1 sec
        Sleep(i);
        continue;
      }
      else
      {
        perror("Cannot open lockfile");
        break;
      }
    }
    break;
  }

  locked_files[num_locked_files].lockfd = lockfd;

  if (lockfd > 0 && i > 0)
  {
    printf("Locked %.250s\n", path);
  }

  f=fopen(path, mode);
  if (f)
  {
    locked_files[num_locked_files++].open_file = f;
  }
  else
  {
    if (close(locked_files[num_locked_files].lockfd) != 0) perror("Failed to close lockfile");
    if (remove(locked_files[num_locked_files].lock_filename)!= 0) perror("Failed to delete lockfile");
  }

  return f;
}

int unlock_and_fclose(FILE *f)
{
  unsigned int i, j;
  int ret;

  if (f == NULL) return -1;

  for (i=0; i<num_locked_files; i++)
  {
    if (locked_files[i].open_file == f)
    {
      ret = fclose(f);
      f = NULL;
      if (close(locked_files[i].lockfd) != 0) perror("Failed to close lockfile");
      if (remove(locked_files[i].lock_filename)!= 0) perror("Failed to delete lockfile");
      for (j=i+1; j<num_locked_files; j++)
      {
        locked_files[j-1].lockfd = locked_files[j].lockfd;
        locked_files[j-1].open_file = locked_files[j].open_file;
        strcpy(locked_files[j-1].lock_filename, locked_files[j].lock_filename);
      }
      num_locked_files--;
      break;
    }
  }
  if (f)
  {
    fprintf(stderr, "File was not locked!\n");
    ret = fclose(f);
  }
  return ret;
}
