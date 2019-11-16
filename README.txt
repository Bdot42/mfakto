** Preface for mfakto 0.15pre6 **

This is a developmental version of mfakto. It has been verified to produce
correct results. However, performance has not been optimized and there may be
bugs and incomplete features. Please help improve mfakto by doing tests,
providing feedback and reporting issues. Of course, code contributions are
always welcome too.

You can get support via the following means:

- the official thread at the GIMPS forum:
  https://mersenneforum.org/showthread.php?t=15646
- opening a ticket on GitHub: https://github.com/Bdot42/mfakto/issues
- contacting Bertram Franz at bertramf@gmx.net

#################
# mfakto README #
#################

Contents

0      What is mfakto?
1      Compilation
1.1    Linux
1.2.1  Windows: MSVC
1.2.2  Windows: MinGW
1.3    macOS
2      Running mfakto
2.1    Supported GPUs
2.2    Linux
2.3    Windows
2.4    macOS
3      Howto get work and report results from/to the primenet server
4      Known issues
4.1    Stuff that looks like an issue but actually isn't an issue
5      Tuning
6      FAQ
7      Plans


#####################
# 0 What is mfakto? #
#####################

mfakto is an OpenCL port of mfaktc that aims to have the same features and
functions. mfaktc is a program that trial factors Mersenne numbers. It stands
for "Mersenne faktorisation* with CUDA" and was written for Nvidia GPUs. mfakto
is used primarily in the Great Internet Mersenne Prime Search.

Primality tests are computationally intensive, but we can save time by finding
small factors. GPUs are very efficient at this task due to their parallel
nature. Only one factor is needed to prove a number composite.

Using a modified Sieve of Eratosthenes, mfakto generates a list of possible
factors for a given Mersenne number. It then uses modular exponentiation to
test these factors. Although this step is only done on the GPU in practice,
mfakto can perform both steps on either the CPU or GPU. You can find more
details at the GIMPS website:
https://mersenne.org/various/math.php#trial_factoring

* portmanteau of the English word "factorisation" and the German word
"Faktorisierung"


#################
# 1 Compilation #
#################

General requirements:
- C and C++ development tools
- an OpenCL SDK

Please note: the AMD APP SDK has been discontinued. If you still decide to use
it to compile mfakto, make sure you have version 2.5 or later. You can download
the AMD APP SDK here: https://community.amd.com/thread/227948

#############
# 1.1 Linux #
#############

Requires:
- ROCm

Steps:
- install ROCm
- navigate to the mfakto folder
- cd src
- verify that the AMD_APP_DIR variable in the makefile points to the SDK
  installation directory
- make
- mfakto should compile without errors in its root folder

#######################
# 1.2.1 Windows: MSVC #
#######################

Requires:
- Microsoft Visual Studio 2012
- GPUOpen OpenCL SDK

Steps:
- install the GPUOpen OpenCL SDK
- launch Visual Studio and open the mfaktoVS12.sln file. You can use a later
  version as Visual Studio will automatically convert your projects. If the
  option does not appear, then right-click the solution and select "Retarget
  solution" from the menu.
- open the project properties and select the configuration and platform
- go to C/C++ > General > Additional Include Directories and add the path to
  the OpenCL headers:

      %OCL_ROOT%\include

- then go to Linker > General > Additional Library Directories and add the path
  to the appropriate library folder:

      32-bit -> %OCL_ROOT%\lib\x86
      64-bit -> %OCL_ROOT%\lib\x86_64

- select Build > Build Solution to compile mfakto

########################
# 1.2.2 Windows: MinGW #
########################

Requires:
- MinGW
- GPUOpen OpenCL SDK
- optional: MSYS2

Initial steps:
- install the GPUOpen OpenCL SDK
- add the "bin" folder in the MinGW directory to your system Path variable
- verify that the AMD_APP_DIR variable in the makefile points to the SDK
  installation directory (see note)

MinGW can be used with or without MSYS to compile mfakto. In the latter case:
- navigate to the mfakto folder
- cd src
- mingw32-make

Otherwise:
- install MSYS2 using the instructions at the home page: https://www.msys2.org
- launch the MSYS2 shell and install the required packages:

      pacman -S mingw-w64-x86_64-gcc make

- launch the 32-bit or 64-bit MinGW shell and navigate to the mfakto folder
- cd src
- make (cross your fingers)

Important note: make does not support spaces in file names. If your OpenCL SDK
installation directory contains spaces, then you will need to either create a
symbolic link or copy the files to another folder.

#############
# 1.3 macOS #
#############

Requires:
- Command Line Tools

Steps:
- cd src
- make -f Makefile.macOS
- mfakto should compile out of the box as macOS contains a native OpenCL
  implementation

####################
# 2 Running mfakto #
####################

General requirements:
- Linux and Windows: AMD Catalyst 11.4 or higher. It is recommended to use
  11.10 or above as the AMD APP SDK has been discontinued.
- otherwise: AMD APP SDK 2.5 or higher

Open a command shell and run 'mfakto -h' in the mfakto folder for parameters it accepts.
You may also want to check mfakto.ini for changing and tweaking mfakto.
Typically you will want to get work from a worktodo file which can be specified in mfakto.ini.

Please run the built-in selftest (mfakto -st) each time you've:
- Recompiled the code
- Downloaded a new binary from somewhere
- Changed the graphics driver
- Changed your hardware

worktodo.txt example:
-- cut here --
Factor=bla,66362159,64,68
Factor=bla,3321932839,50,61
-- cut here --

Then run 'mfakto'. If everything is working as expected this should trial
factor M66362159 from 2^64 to 2^68 and after that trial factor
M3321932839 from 2^50 to 2^61.

######################
# 2.1 Supported GPUs #
######################

  AMD:
- R9 xxx, R7 xxx, R5 xxx
- HD7xxx, HD8xxx
- HD5xxx, HD6xxx, including the builtin HD6xxx on AMD APUs
- HD4xxx, FireStream 92xx (no atomic operations available) *
- not supported: (kernel compilation fails): HD2xxx, HD3xxx, FireStream 91xx

* without atomics, reporting multiple factors found in the same block/grid
will not work. Tests showed that only one of the factors will be reported,
but theoretically it could happen that even the reported factor is incorrect
(due to consisting of a mix of bytes of multiple factors). In cases when
mfakto reports a factor but the factor is incorrect (rejected by primenet),
please rerun the test of the exponent and the bitlevel on the CPU (e.g.
prime95 or mfakto -d c).

#############
# 2.2 Linux #
#############

- build mfakto using the above instructions
- run mfakto

###############
# 2.3 Windows #
###############

- AMD Catalyst 11.4 is the minimum required version. It is recommended to use
  11.10 or above as the AMD APP SDK has been discontinued.
- otherwise: AMD APP SDK 2.5 or higher. In this case, make sure the path to the
  appropriate library folder is in the system Path variable:

      32-bit -> %AMDAPPSDKROOT%\lib\x86
      64-bit -> %AMDAPPSDKROOT%\lib\x86_64

- you may need the Microsoft Visual C++ 2010 Redistributable Package for your
  platform and language:

      32-bit -> https://microsoft.com/en-us/download/details.aspx?id=5555
      64-bit -> https://microsoft.com/en-us/download/details.aspx?id=14632

- only the 64-bit binary is currently available.

#############
# 2.4 macOS #
#############

- build mfakto using the above instructions
- mfakto should run without any additional software

####################################################################
# 3 How to get work and report results from/to the primenet server #
####################################################################

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

Start mfakto and stress your GPU! ;)

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
    When you increase the upper limit of your assignments it is important to
    report the results once you've finished up to the desired level. (Do not
    report partially results before!)


##################
# 4 Known issues #
##################

- On HD77xx, 78xx, 79xx and R series, mfakto may run very slow at 99% GPU load.
  mfakto warns about the issue during startup.
  The reason is because of the lower number of registers available to the kernels.
  Set VectorSize=2 in mfakto.ini and restart mfakto. It should be better now.
- The user interface is not hardened against malformed input. There are some
  checks but if you really try you should be able to screw it up.
- The GUI of your OS may be very laggy while running mfakto. In severe
  cases, if a single kernel invocation takes too long, Windows may decide
  the driver is faulty and reboot.
  Try lowering GridSize in mfakto.ini. Smaller grids should have better
  responsiveness at a little performance penalty. Performancewise this is not
  recommended on GPUs which can handle well over 100M/s candidates.
  If that does not help, try lowering NumStreams to 2 or even 1.
- SievePrimesAdjust works now, but is not always optimal. Test it out and
  see what the best SievePrimes is, set it and fix it by setting
  SievePrimesAdjust to 0.
- GPU is not found, fallback to CPU
  This happens on Linux when there is no X-server running, or the X-server
  is not accessible. It happens on Windows when not connected to the primay
  display (e.g. being connected through terminal services). So please try to
  run mfakto locally on the main X-display. If that fails as well or is not the case,
  then the graphics driver may be too old. Also, check the output of clinfo
  for your GPU. If the drivers are up to date, then maybe
  your AMD GPU is not the first GPU. Try the -d switch to specify a different
  device number.

##################################################################
# 4.1 Stuff that looks like an issue but actually isn't an issue #
##################################################################

- mfakto runs slower on small ranges. Usually it doesn't make much sense to
  run mfakto with an upper limit smaller than 2^64. It is designed for trial
  factoring above 2^64 up to 2^92 (factor sizes). ==> mfakto needs
  "long runs"!
- mfakto can find factors outside the given range.
  E.g. './mfakto.exe -tf 66362159 40 41' has a high change to report
  124246422648815633 as a factor. Actually this is a factor of M66362159 but
  it's size is between 2^56 and 2^57! Of course
  './mfakto.exe -tf 66362159 56 57' will find this factor, too. The reason
  for this behaviour is that mfakto works on huge factor blocks. This is
  controlled by GridSize in mfakto.ini. The default value is 3 which means
  that mfakto runs up to 1048576 factor candidates at once (per class). So
  the last block of each class is filled up with factor candidates above the
  upper limit. While this is a huge overhead for small ranges it's safe to
  ignore it on bigger ranges. If a class contains 100 blocks the overhead is
  on average 0.5%. When a class needs 1000 blocks the overhead is 0.05%...


############
# 5 Tuning #
############

You can find additional settings in the mfakto.ini file. Read it carefully
before making changes. ;-)


#########
# 6 FAQ #
#########

Q Does mfakto support multiple GPUs?
A No, but using the commandline option "-d <GPU number>" you should
  be able to specify which GPU to use for each specific mfakto instance.
  Please read the next question, too.

Q Can I run multiple instances of mfakto on the same computer?
A Yes, and in most cases this is necessary to make full use of the GPU(s) if sieving with CPU.
  If the sieve is running on the GPU(default), one instance should fully utilize
  a single GPU.

Q Which tasks should I assign to mfakto?
A Currently, the 73-bit-barrett kernel is the fastest one, working for factors
  from 60 bits to 73 bits. Selecting tasks for this kernel will give best
  results. The 79-bit-barrett kernel is quite fast too.

Q I modified something in the kernel files but my changes are not picked up by
  mfakto. Why not?
A Since mfakto version 0.14, mfakto tries to load precompiled kernel files.
  The ini-file parameter UseBinfile (default: mfakto_Kernels.elf) defines the
  file name of the precompiled kernels. Delete the file and restart mfakto, it
  will then compile the kernels from the source files.


###########
# 7 Plans #
###########

- keep features/changes in sync with mfaktc
- performance improvements whenever I find them ;)
- documentation and comments in code
- full 95-bit implementation
- perftest modes for kernel speed.
- build a GCN-assembler-kernel
