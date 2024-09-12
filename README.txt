** Preface for mfakto 0.16-beta.1 **

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
3      Getting work and reporting results
4      Known issues
4.1    Non-issues
5      Tuning
6      FAQ
7      Plans


#####################
# 0 What is mfakto? #
#####################

mfakto is an OpenCL port of mfaktc that aims to have the same features and
functions. mfaktc is a program that trial factors Mersenne numbers. It stands
for "Mersenne faktorisation* with CUDA" and was written for Nvidia GPUs. Both
programs are used primarily in the Great Internet Mersenne Prime Search.

Primality tests are computationally intensive, but we can save time by finding
small factors. GPUs are very efficient at this task due to their parallel
nature. Only one factor is needed to prove a number composite.

mfakto uses a modified Sieve of Eratosthenes to generate a list of possible
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

Please note: the AMD APP SDK has been discontinued. If you still want to use it
to compile mfakto, make sure you have version 2.5 or later. You can download
the SDK here: https://community.amd.com/thread/227948

#############
# 1.1 Linux #
#############

Requires:
- ROCm

Steps:
- install ROCm
- navigate to the mfakto folder
- cd src
- verify that the AMD_APP_DIR variable in the makefile points to the ROCm
  directory
- make
- mfakto should compile without errors in its root folder

#######################
# 1.2.1 Windows: MSVC #
#######################

Requires:
- Microsoft Visual Studio
- an OpenCL SDK

Steps:
- download and install the OpenCL SDK from AMD:
  https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases
- open mfaktoVS12.sln in Visual Studio. You can use any recent version as
  Visual Studio will automatically update your project settings. If the option
  does not appear, right-click the solution and select "Retarget solution" from
  the menu.
- open the project properties and select the configuration and platform
- go to C/C++ > General > Additional Include Directories and ensure it contains
  the path to the OpenCL headers:

      $(OCL_ROOT)\include

- now go to Linker > General > Additional Library Directories and ensure it
  contains the correct paths:

      32 bits: $(OCL_ROOT)\lib\x86
      64 bits: $(OCL_ROOT)\lib\x86_64

- select Build > Build Solution to compile mfakto

Additional notes:
- As an alternative, you can use the OpenCL SDK from the Khronos Group instead:
  https://github.com/KhronosGroup/OpenCL-SDK/releases
- Please note the AMD installer automatically sets the OCL_ROOT environment
  variable. Developers using a different SDK will have to manually configure
  the library and include paths in the project properties.
- You may have to relaunch Visual Studio before it recognizes changes to
  environment variables.

########################
# 1.2.2 Windows: MinGW #
########################

Requires:
- MinGW (64-bit)
- an OpenCL SDK
- optional: MSYS2

Initial steps:
- download and install a 64-bit MinGW compiler. Our recommendation is to use
  MinGW-w64 as it is actively maintained: http://mingw-w64.org
- download and install the OpenCL SDK from AMD:
  https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases
- add the "bin" folder in the MinGW directory to your system Path variable
- verify that the AMD_APP_DIR variable in the makefile points to the SDK
  directory (see note)

MinGW can be optionally used with MSYS2 to compile mfakto:
- install MSYS2 using the instructions at the home page: https://www.msys2.org
- launch the MSYS2 shell and install the required packages:

      pacman -S mingw-w64-x86_64-gcc make

- start the 32-bit or 64-bit MinGW shell and navigate to the mfakto folder
- cd src
- make (cross your fingers)

Otherwise:
- navigate to the mfakto folder
- cd src
- mingw32-make

Additional notes:
- You may see some warnings, but they are safe to ignore.
- make does not support spaces in file names. If the path to the OpenCL SDK
  contains spaces, then you will need to either create a symbolic link or copy
  the files to another folder.
- mfakto may not compile with Win-builds. It is recommended to use the native
  Windows package instead: https://sourceforge.net/projects/mingw-w64/files
- To compile mfakto for both 32 and 64 bits, you will need to install MinGW-w64
  for both the i686 and x86_64 architectures.
- mfakto may give an "entry point not found" error on startup. Running make
  with the "static=yes" flag should prevent this.

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
- AMD Catalyst 11.4 or higher. Consider using at least 14.4 as some previous
  versions have a bug that causes high CPU loads.
- AMD APP SDK 2.5 or higher for systems without Catalyst 11.10 or above. It is
  recommended to update your drivers as the SDK has been discontinued.
- for Intel integrated GPUs: Compute Runtime for OpenCL

macOS users do not need any additional software as OpenCL is already part of
the system.

Open a terminal window and run 'mfakto -h' for possible parameters. You may
also want to check mfakto.ini for additional settings. mfakto typically fetches
work from worktodo.txt as specified in the INI file. See section 3 on how to
obtain assignments and report results.

A typical worktodo.txt file looks like this:
  -- begin example --
  Factor=[assignment ID],66362159,64,68
  Factor=[assignment ID],3321932899,76,77
  -- end example --

You can launch mfakto after getting assignments. In this case, mfakto should
trial factor M66362159 from 64 to 68 bits, followed by M3321932899 from 76 to
77 bits.

mfakto has a built-in self-test that automatically optimizes parameters. Please
run 'mfakto -st' each time you:
- Recompile the code
- Download a new binary from somewhere
- Change the graphics driver
- Change your hardware

######################
# 2.1 Supported GPUs #
######################

AMD:
- all devices that support OpenCL 1.1 or later
- all APUs
- OpenCL 1.0 devices, such as the FireStream 9250 / 9270 and Radeon HD 4000
  series, can run mfakto but do not support atomic operations*
- not supported: FireStream 9170 and Radeon HD 2000 / 3000 series (as kernel
  compilation fails)

Other devices:
- Intel HD Graphics 4000 and later. Currently not supported on macOS.
- OpenCL-enabled CPUs via the '-d c' option. Currently fails
- Nvidia devices. Supported but may fail on some hardware


* without atomics, mfakto may not correctly process multiple factors found in
the same class. It may report only one factor or even an incorrect one, the
latter due to scrambled data from multiple factors. PrimeNet automatically
rejects factors that do not divide a Mersenne number. If this happens, run the
exponent and bit level again on the CPU or another device. You can run mfakto
on the CPU using the '-d c' option or use Prime95 instead. Lowering GridSize in
mfakto.ini can also reduce the chance of error.

#############
# 2.2 Linux #
#############

- build mfakto using the above instructions
- run mfakto

###############
# 2.3 Windows #
###############

Requirements:
- AMD Catalyst 11.4 or higher. Consider using at least 14.4 as some previous
  versions have a bug that causes high CPU loads.
- AMD APP SDK 2.5 or higher for systems without Catalyst 11.10 or above. It is
  recommended to update your drivers as the SDK has been discontinued.
  If you still want to use it to run mfakto, make sure the path to the
  appropriate library folder is in the system Path variable:

      32 bits: %AMDAPPSDKROOT%\lib\x86
      64 bits: %AMDAPPSDKROOT%\lib\x86_64

- you may also need the Microsoft Visual C++ 2010 Redistributable Package for
  your platform and language:

      32 bits: https://microsoft.com/en-us/download/details.aspx?id=5555
      64 bits: https://microsoft.com/en-us/download/details.aspx?id=14632

Steps:
- build mfakto using the above instructions or download a stable version. Only
  the 64-bit binary is currently distributed.
- go to the mfakto folder and launch the executable
- mfakto defaults to the first OpenCL-supported GPU it finds. You can specify a
  different GPU using the -d option.

#############
# 2.4 macOS #
#############

- build mfakto using the above instructions
- mfakto should run without any additional software

########################################
# 3 Getting work and reporting results #
########################################

You must have a PrimeNet account to participate. Simply visit the GIMPS website
at https://mersenne.org to create one. Once you've signed up, you can get
assignments in several ways.

From the GIMPS website:
    Step 1) log in to the GIMPS website with your username and password
    Step 2) on the menu bar, select Manual Testing > Assignments
    Step 3) open the link to the manual GPU assignment request form
    Step 4) enter the number of assignments or GHz-days you want
    Step 5) click "Get Assignments"

    Users with older GPUs may want to use the regular form.

Using the GPU to 72 tool:
    GPU to 72 is a website that "subcontracts" assignments from the PrimeNet
    server. It was previously the only means to obtain work at high bit levels.
    Although the manual GPU assignment form now serves this purpose, GPU to 72
    remains the more popular option.

    GPU to 72 website: https://gpu72.com

Using the MISFIT tool:
    MISFIT is a Windows tool that automatically requests assignments and
    submits results. You can get it here: https://mersenneforum.org/misfit

From mersenne.ca:
    James Heinrich's website mersenne.ca offers assignments for exponents up
    to 32 bits. You can get such work here: https://mersenne.ca/tf1G

    Be aware that mfakto currently does not work below 60 bits.

Advanced usage:
    As mfakto works best on long-running jobs, you may want to manually extend
    your assignments. Let's assume you've received an assignment like this:
        Factor=[assignment ID],78467119,65,66

    This means the PrimeNet server has assigned you to trial factor M78467119
    from 65 to 66 bits. However, take a look at the factoring limits:
    http://mersenne.org/various/math.php

    According to the table, the exponent is factored to 71 bits before being
    tested. Because mfakto runs very fast on modern GPUs, you might want to go
    directly to 71 or even 72 bits. Simply edit the ending bit level before
    starting mfakto. For example:
        Factor=[assignment ID],78467119,65,72

    It is important to submit the results once you're done. Do not report
    partial results as the exponent may be reassigned to someone else in the
    interim, resulting in duplicate work and wasted cycles.

    Please do not manually extend assignments from GPU to 72 as users are
    requested not to "trial factor past the level you've pledged."


    Once you have your assignments, copy the "Factor=..." lines directly into
    your worktodo.txt file. Start mfakto, sit back and let it do its job.
    Running mfakto is also a great way to stress test your GPU. ;-)

Submitting results:
    mfakto currently cannot communicate with the PrimeNet server, so you must
    manually submit the results. To prevent abuse, admin approval is required
    for manual submissions. You can request approval by contacting George
    Woltman at woltman@alum.mit.edu or posting on the GIMPS forum:
    https://mersenneforum.org/forumdisplay.php?f=38

    Step 1) log in to the GIMPS website with your username and password
    Step 2) on the menu bar, select Manual Testing > Results
    Step 3) upload the results.txt file produced by mfakto. You may archive or
            delete the file after it has been processed.

    There are several tools that can automate this process. You can find a
    complete list here:
    https://mersenneforum.org/showpost.php?p=465293&postcount=24


##################
# 4 Known issues #
##################

- On some devices, such as the Radeon HD 7700 - 7900 series, mfakto may be very
  slow at full GPU load due to fewer registers being available to the kernels.
  It will warn about this during startup.
  Set VectorSize=2 in mfakto.ini and restart mfakto to resolve this.

- The user interface has not been extensively tested against invalid inputs.
  Although there are some checks, they are not foolproof by any means.

- Your GUI may lag while running mfakto. On some Windows systems, the OS may
  restart the driver or even throw a BSoD in severe cases.
  Try lowering GridSize or NumStreams in your mfakto.ini file. Smaller grids
  should have better responsiveness at a slight performance loss. Another
  option for Windows users is to increase the GPU processing time:
  https://support.microsoft.com/en-us/help/2665946

- SievePrimesAdjust is not always optimal. Experiment to find the best
  SievePrimes value and set SievePrimesAdjust=0 in your mfakto.ini file.

- GPU is not found, fallback to CPU
  This happens on Linux when there is no X server. It can also happen on
  Windows when the GPU is not the primary display adapter. Try running mfakto
  on the main display rather than remotely. If that fails, then your graphics
  driver may be too old. It's also possible that the first device is not an
  AMD GPU. In this case, use the -d switch to specify a different device
  number. You can run 'clinfo' to get a list of devices.

- on devices that do not support atomic operations, mfakto may not correctly
  process multiple factors found in the same class. It may report only one
  factor or even an incorrect one, the latter due to scrambled data from
  multiple factors.
  If this happens, run the exponent and bit level again on the CPU or another
  device. You can tell mfakto to run on the CPU using the '-d c' option or use
  Prime95 instead. Lowering GridSize in mfakto.ini can also reduce the chance
  of error.

- mfakto does not support Intel HD Graphics on macOS
  Due to buggy drivers shipped with macOS, mfakto presently does not work with
  Intel HD Graphics. Unless Apple fixes the issue, Intel integrated GPUs may
  not be supported in the foreseeable future.


##################
# 4.1 Non-issues #
##################

- mfakto runs slower on small ranges. Usually it doesn't make much sense to
  run mfakto with an upper limit below 64 bits. mfakto is designed to find
  factors between 64 and 92 bits, and is best suited for long-running jobs.

- mfakto can find factors outside the given range.
  This is because mfakto works on huge factor blocks, controlled by GridSize in
  the INI file. The default value GridSize=3 means mfakto runs up to 1048576
  factor candidates at once, per class. So the last block of each class is
  filled with factor candidates above the upper limit. This is a huge overhead
  for small ranges but can be safely ignored for larger ranges. For example,
  the average overhead is 0.5% for a class with 100 blocks but only 0.05% for
  one with 1000 blocks.


############
# 5 Tuning #
############

You can find additional settings in the mfakto.ini file. Read it carefully
before making changes. ;-)


#########
# 6 FAQ #
#########

Q: Does mfakto support multiple GPUs?
A: Currently no, but you can use the -d option to start an instance on a
   specific device. Please also see the next question.

Q: Can I run multiple instances of mfakto on the same computer?
A: Yes. In most cases, this is necessary to make full use of a GPU when sieving
   on the CPU. Otherwise, one instance should fully utilize a single GPU.

Q: What tasks should I assign to mfakto?
A: The 73-bit Barrett kernel is currently the fastest and works for factors
   between 60 to 73 bits. Selecting tasks for this kernel will give best
   results. However, the 79-bit Barrett kernel is quite fast too.

Q: I modified something in the kernel files, but my changes are not picked up
   by mfakto. How come?
A: mfakto tries to load the pre-compiled kernel files in version 0.14 and
   later. The INI file parameter UseBinfile defines the name of the file
   containing the pre-compiled kernels. You can force mfakto to recompile the
   kernels by deleting the file and restarting mfakto.


###########
# 7 Plans #
###########

- keep features/changes in sync with mfaktc
- performance improvements whenever I find them ;)
- documentation and comments in code
- full 95-bit implementation
- perftest modes for kernel speed.
- build a GCN-assembler-kernel
