#################
# mfakto README #
#################

Content

0   What is mfakto?
1   Compilation
1.1 Compilation (Linux)
1.2 Compilation (Windows)
2   Running mfakto
2.1 Supported GPUs
2.2 Running mfakto (Linux)
2.3 Running mfakto (Windows)
3   Howto get work and report results from/to the primenet server
4   Known issues
4.1 Stuff that looks like an issue but actually isn't an issue
5   Tuning
6   FAQ
7   .plan



####################
# 0 What is mfakto #
####################

mfakto is the OpenCL-port of mfaktc. It aims to contain the same features
and application use cases in time.
mfaktc is a program for trial factoring of mersenne numbers. The name mfaktc
is "Mersenne FAKTorisation with Cuda". Faktorisation is a mixture of the
English word "factorisation" and the German word "Faktorisierung".
It can use CPU and GPU resources.



#################
# 1 Compilation #
#################

 AMD APP 2.5 or above is required


###########################
# 1.1 Compilation (Linux) #
###########################

- Install AMD APP >= 2.5
- cd src
- edit Makefile, set the AMD_APP_DIR location
- make


#############################
# 1.2 Compilation (Windows) #
#############################

- Install AMD APP >= 2.5
- Use the VS2010 solution to build the 32-bit or 64-bit binary, or
- use the Linux Makefile as an example how to build your own Windows Makefile


####################
# 2 Running mfakto #
####################

Install Catalyst driver, version >= 11.4
Install AMD APP version >= 2.5 (not required for Catalyst 11.10 or above)

Catalyst driver 11.9 uses up to one CPU core less than its predecessors:
11.9 strongly recommended.

The high CPU usage bug is back in Catalyst 13.4 and 13.5 - stay below.

Open a command shell and run 'mfakto -h'. It will tell you what parameters
it accepts. Maybe you want to tweak the parameters in mfakto.ini. A short
description of those parameters is included in mfakto.ini, too.
Typically you want to get work from a worktodo file. You can specify the
name in mfakto.ini. It was tested with primenet v5 worktodo files but v4
should work, too.

Please run the builtin selftest (mfakto -st) each time you've
- recompiled the code
- downloaded a new binary from somewhere
- changed the graphics driver
- changed your hardware

Example worktodo.txt
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

- HD7xxx
- HD5xxx, HD6xxx, including the builtin HD6xxx on AMD APUs
- HD4xxx, FireStream 92xx (no atomic operations available *
- not supported (kernel compilation fails): HD2xxx, HD3xxx, FireStream 91xx

* without atomics, reporting multiple factors found in the same block/grid
will not work. Tests showed that only one of the factors will be reported, 
but theoretically it could happen that even the reported factor is incorrect
(due to consisting of a mix of bytes of multiple factors). In cases when
mfakto reports a factor but the factor is incorrect (rejected by primenet),
please rerun the test of the exponent and the bitlevel on the CPU (e.g.
prime95 or mfakto -d c).

##############################
# 2.2 Running mfakto (Linux) #
##############################

- AMD APP 2.5 or higher and Catalyst driver 11.4 or higher are required
- run mfakto
- precompiled version is only available for 64-bit (built on SuSE 11.4)

################################
# 2.3 Running mfakto (Windows) #
################################

- Catalyst driver 11.4 or higher are required
- if driver < 11.10, install AMD APP SDK 2.5 and make sure
  $AMD_APP_DIR/lib/x86_64 is in the path
- Microsoft Visual C++ 2010 Redistributable Package for your platform and
  language, e.g.
  http://www.microsoft.com/downloads/details.aspx?familyid=BD512D9E-43C8-4655-81BF-9350143D5867&displaylang=de
- 32-bit and 64-bit precompiled versions available; 64-bit siever is way more
  efficient

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

Start mfakto and stress your GPU ;)

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

- On HD77xx, 78xx and 79xx, mfakto may run awfully slow at 99% GPU load.
  mfakto warns about the issue during startup.
  The reason is the lower number of registers available to the kernels.
  Set VectorSize=2 in mfakto.ini and restart mfakto. It should be fast now.
- Even with GPU sieving enabled, mfakto uses up to a CPU core. This is a bug
  in the Catalyst 13.4 and 13.5 drivers, hopefully fixed soon. Downgrade
  drivers to 13.3, if possible.
- The user interface is not hardened against malformed input. There are some
  checks but when you really try you should be able to screw it up.
- The GUI of your OS might be very laggy while running mfakto. In severe
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
  run mfakto locally on the main X-display. If that fails as well, then the
  graphics driver may be too old. Check the clinfo (part of AMD APP SDK)
  output for your GPU. If drivers and AMD APP SDK are up to date, then maybe
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
  upper limit. While this is a huge overhead for small ranges it's save to
  ignore it on bigger ranges. If a class contains 100 blocks the overhead is
  on average 0.5%. When a class needs 1000 blocks the overhead is 0.05%...



############
# 5 Tuning #
############

Read mfakto.ini and think before editing. ;)



#########
# 6 FAQ #
#########

Q Does mfakto support multiple GPUs?
A Not tested yet, but using the commandline option "-d <GPU number>" you should
  be able to specify which GPU to use for each specific mfakto instance.
  Please read the next question, too.

Q Can I run multiple instances of mfakto on the same computer?
A Yes, and in most cases this is necessary to make full use of the GPU(s) only if
  CPU sieving. If the sieve is running on the GPU(default), one instance should fully utilize
  a single GPU.
  

Q Which tasks should I assign to mfakto?
A Currently, the 73-bit-barrett kernel is the fastest one, working for factors
  from 60 bits to 73 bits. Selecting tasks for this kernel will give best
  results. The 79-bit-barrett kernel is quite fast too.


###########
# 7 .plan #
###########

- keep features/changes in sync with mfaktc
- performance improvements whenever I find them ;)
- documentation and comments in code
- full 95-bit implementation
- Makefile.win  --  any volunteers?
- perftest modes for kernel speed.
- retrieve L1/L2-cache-size and optimize sieve accordingly at runtime
- Implement&test AllowSleep
