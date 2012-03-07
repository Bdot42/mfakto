#################
# mfakto README #
#################

Content

0   What is mfakto?
1   Compilation
1.1 Compilation (Linux)
1.2 Compilation (Windows)
2   Running mfakto (Linux)
2.1 Running mfakto (Windows)
3   Howto get work and report results from/to the primenet server
4   Known issues
4.1 Stuff that looks like an issue but actually isn't an issue
5   Tuning
6   FAQ
7   .plan



####################
# 0 What is mfakto #
####################

mfakto is a program for trial factoring of mersenne numbers. The name mfakto
is "Mersenne FAKTorisation with OpenCL". Faktorisation is a mixture of the
english word "factorisation" and the german word "Faktorisierung".
It uses CPU and GPU resources.



#################
# 1 Compilation #
#################

* To be done adjusted *

It is assumed that you've allready setup your compiler and AMD APP environment.
There are some compiletime settings in the file src/params.h possible:
- in the upper part of the file there are some settings which "advanced
  users" can chance if they think it is beneficial. Those settings are
  verified for reasonable values.
- in the middle are some debug options which can be turned on. These options
  are only usefull for debuging purposes.
- the third part contains some defines which should _NOT_ be changed unless
  you really know what they do. It is easily possible to screw up something.

A 64bit built is prefered except for some old lowend GPUs because the
performance critical CPU code runs ~33% faster compared to 32bit. (measured
on a Intel Core i7)



###########################
# 1.1 Compilation (Linux) #
###########################

Change into the subdirectory "src/"

Adjust the path to your CUDA installation in "Makefile" and run 'make'.
The binary "mfakto.exe" is placed into the parent directory.

I'm using
- OpenSUSE 11.1 x86_64
- gcc 4.3.2 (OpenSUSE 11.1)
- Nvidia driver 260.24
- Nvidia CUDA Toolkit
  - 4.0 RC2   read below
  - 3.2       ~1% slower than 3.0/3.1
  - 3.1       read below
  - 3.0       read below

CUDA Toolkits 3.0, 3.1 an 4.0RC2 are only basically tested (compile, run
selftest).

I don't spent time testing mfakto on 32bit Linux because I think 64bit
(x86_64) is adopted by most Linux users now. Anyway mfakto should work on
32bit Linux, too. If problems are reported I'll try to fix them. So I don't
drop Linux 32bit support totally. ;)

When you compile mfakto on a 32bit system you must change the library path
in "Makefile" (replace "lib64" with "lib").



#############################
# 1.2 Compilation (Windows) #
#############################

The following instructions have been tested on Windows 7 64bit using Visual
Studio 2008 Professional. A GNU compatible version of make is also required
as the Makefile is not compatible with nmake. GNU Make for Win32 can be
downloaded from http://gnuwin32.sourceforge.net/packages/make.htm.

Run the Visual Studio 2008 x64 Win64 Command Prompt and change into the
"src/" subdirectory.

Run 'make -f Makefile.win' for a 64bit built (recommended on 64bit systems)
or 'make -f Makefile.win32' for a 32bit built. Perhaps you have to adjust
the paths to your CUDA installation and the Microsoft Visual Studio binaries
in the makefiles. The binaries "mfakto-win-64.exe" or "mfakto-win-32.exe"
are placed in the parent directory.



############################
# 2 Running mfakto (Linux) #
############################

* Not yet available *

Just run './mfakto.exe -h'. It will tell you what parameters it accepts.
Maybe you want to tweak the parameters in mfakto.ini. A small describtion
of those parameters is included in mfakto.ini, too.
Typically you want to get work from a worktodo file. You can specify the
name in mfakto.ini. It was tested with primenet v5 worktodo files but v4
should work, too.

Please run the builtin selftest each time you've
- recompiled the code
- downloaded a new binary from somewhere
- changed the Nvidia driver
- changed your hardware

Example worktodo.txt
-- cut here --
Factor=bla,66362159,64,68
Factor=bla,3321932839,50,71
-- cut here --

Than run e.g. './mfakto.exe'. If everything is working as expected this
should trial factor M66362159 from 2^64 to 2^68 and after that trial factor
M3321932839 from 2^50 to 2^71.



################################
# 2.1 Running mfakto (Windows) #
################################

Similar to Linux (read above!).
Open a command shell and run 'mfakto.exe -h'.
To run the selftest: 'mfakto -st'
To get details about the OpenCL env: 'mfakto --CLtest'


###################################################################
# 3 Howto get work and report results from/to the primenet server #
###################################################################

Getting work:
    Step 1) go to http://www.mersenne.org/ and login with your username and
            password
    Step 2) on the menu on the left click "Manual Testing" and than
            "Assignments"
    Step 3) choose the number of assignments by choosing
            "Number of CPUs (cores) you need assignments for (maximum 12)"
            and "Number of assignments you want for each core"
    Step 4) Change "Preferred work type" to "Trial factoring"
    Step 5) click the button "Get Assignments"
    Step 6) copy&paste the "Factor=..." lines directly into the worktodo.txt
            in your mfakto directory

Start mfakto and stress your GPU ;)

Once mfakto has finished all the work report the results to the primenet
server:
    Step 1) go to http://www.mersenne.org/ and login with your username and
            password
    Step 2) on the menu on the left click "Manual Testing" and than
            "Results"
    Step 3) upload the results.txt file generated by mfakto using the
            "search" and "upload" button
    Step 49 once you've verified that the primenet server has recognized
            your results delete or rename the results.txt from mfakto

Advanced usage (extend the upper limit):
    Since mfakto works best on long running jobs you may want to extend the
    upper TF limit of your assignments a little bit. Take a look how much TF
    is usually done here: http://www.mersenne.org/various/math.php
    Lets assume that you've received an assignment like this:
        Factor=<some hex key>,78467119,65,66
    This means that primenet server assigned you to TF M78467119 from 2^65
    to 2^66. Take a look at the site noted above, those exponent should be
    TFed up to 2^71. Primenet will do this in multiple assignments (step by
    step) but since mfakto runs very fast on modern GPUs you might want to
    TF up to 2^71 or even 2^72 directly. Just replace the 66 at the end of
    the line with e.g. 72 before you start mfakto:
        e.g. Factor=<some hex key>,78467119,65,72
    When you increase the upper limit of your assignments it is import to
    report the results once you've finished up to the desired level. (Do not
    report partially results before!)



##################
# 4 Known issues #
##################

- The user interface isn't hardened against malformed input. There are some
  checks but when you really try you should be able to screw it up.
- The GUI of your OS might be very laggy while running mfakto. (newer GPUs
  with compute capabilty 2.0 or higher can handle this _MUCH_ better)
  Comment from James Heinrich:
    Slower/older GPUs (e.g. compute v1.1) that experience noticeable lag can
    get a significant boost in system usability by reducing the NumStreams
    setting from default "3" to "2", with minimal performance loss.
    Decreasing to "1" provides much greater system responsiveness, but also
    much lower throughput.
    At least it did so for me. With NumStreams=3, I could only run mfakto
    when I wasn't using the computer. Now I run it all the time (except when
    watching a movie or playing a game...)
  Another try worth are different settings of GridSize in mfakto.ini.
  Smaller grids should have higher responsibility with the cost of a little
  performance penalty. Performancewise this is not recommended on GPUs which
  can handle >= 100M/s candidates.
- the debug options CHECKS_MODBASECASE (and USE_DEVICE_PRINTF) might report
  too high qi values while using the barrett kernels. They are caused by
  factor candidates out of the specified range.



##################################################################
# 4.1 Stuff that looks like an issue but actually isn't an issue #
##################################################################

- mfakto runs slower on small ranges. Usually it doesn't make much sense to
  run mfakto with an upper limit smaller than 2^64. It is designed for trial
  factoring above 2^64 up to 2^95 (factor sizes). ==> mfakto needs
  "long runs"!
- mfakto can find factors outside the given range.
  E.g. './mfakto.exe -tf 66362159 40 41' has a high change to report
  124246422648815633 as a factor. Actually this is a factor of M66362159 but
  it's size is between 2^56 and 2^57! Offcourse
  './mfakto.exe -tf 66362159 56 57' will find this factor, too. The reason
  for this behaviour is that mfakto works on huge factor blocks. This is
  controlled by GridSize in mfakto.ini. The default value is 3 which means
  that mfakto runs up to 1048576 factor candidates at once (per class). So
  the last block of each class is filled up with factor candidates above to
  upper limit. While this is a huge overhead for small ranges it's save to
  ignore it on bigger ranges. If a class contains 100 blocks the overhead is
  on average 0.5%. When a class needs 1000 blocks the overhead is 0.05%...



############
# 5 Tuning #
############

Read mfakto.ini and think before edit. ;)


