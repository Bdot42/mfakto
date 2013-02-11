/*
This file is part of mfaktc (mfakto).
Copyright (C) 2009 - 2011  Oliver Weihe (o.weihe@t-online.de)
This file has been written by Luigi Morelli (L.Morelli@mclink.it)

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

#if !defined(TRUE)	// keep self-contained
	#define FALSE (0)
	#define TRUE (1)
#endif

enum ASSIGNMENT_ERRORS
{	NEVER_ASSIGNED=-1,
	OK=0,
	CANT_OPEN_FILE=1,
	VALID_ASSIGNMENT_NOT_FOUND=2,
	CANT_OPEN_WORKFILE=3,
	CANT_OPEN_TEMPFILE=4,
	ASSIGNMENT_NOT_FOUND=5,
	CANT_RENAME =6
};
#define MAX_LINE_LENGTH 100
typedef char LINE_BUFFER[MAX_LINE_LENGTH+1];

struct ASSIGNMENT
{
	unsigned int exponent;
	int bit_min;
	int bit_max;
	char assignment_key[MAX_LINE_LENGTH+1];	// optional assignment key....
	char comment[MAX_LINE_LENGTH+1];	// optional comment.
						// if it has a newline at the end, it was on a line by itself preceding the assignment.
						// otherwise, it followed the assignment on the same line.
};


int valid_assignment(unsigned int exp, int bit_min, int bit_max, int verbosity);	// nonzero if assignment is valid
enum ASSIGNMENT_ERRORS get_next_assignment(char *filename, unsigned int *exponent, int *bit_min, int *bit_max, LINE_BUFFER *assignment_key, int verbosity);
enum ASSIGNMENT_ERRORS clear_assignment(char *filename, unsigned int exponent, int bit_min, int bit_max, int bit_min_new);

int add_file_available(char *filename);

/* process the add file for the worktodo file <filename> */
int process_add_file(char *filename);