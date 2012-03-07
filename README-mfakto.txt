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

mfakto is the OpenCL-port of mfaktc. It aims to contain the same features
and application use cases in time.
mfaktc is a program for trial factoring of mersenne numbers. The name mfaktc
is "Mersenne FAKTorisation with Cuda". Faktorisation is a mixture of the
english word "factorisation" and the german word "Faktorisierung".
It uses CPU and GPU resources.



#################
# 1 Compilation #
#################

*** not yet applicable to mfakto, only precompiled Win64 available ***


###########################
# 1.1 Compilation (Linux) #
###########################



#############################
# 1.2 Compilation (Windows) #
#############################



############################
# 2 Running mfakto (Linux) #
############################

*** not yet applicable to mfakto, only precompiled Win64 available ***


################################
# 2.1 Running mfakto (Windows) #
################################

Open a command shell and run 'mfakto.exe -h'. It will tell you what parameters
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
Factor=bla,3321932839,50,71
-- cut here --

Then run e.g. 'mfakto.exe'. If everything is working as expected this
should trial factor M66362159 from 2^64 to 2^68 and after that trial factor
M3321932839 from 2^50 to 2^71.



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

DO NOT YET REPORT ANY MFAKTO RESULTS TO PRIMENET

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
- The GUI of your OS might be very laggy while running mfakto. In severe
  cases, if a single kernel invocation takes too long, Windows may decide
  the driver is faulty and reboot.
  Try lowering GridSize in mfakto.ini. Smaller grids should have better
  responsiveness at a little performance penalty. Performancewise this is not
  recommended on GPUs which can handle >= 100M/s candidates.
  If that does not help, try lowering NumStreams.
- SievePrimesAdjust does not yet work (it will quickly bring you to the 200000
  limit, no matter what), better leave it at 0.
- Sometimes, when multiple instances of mfakto are running and one exits, the
  whole machine locks up - hard reboot required. Reason unknown.
  This did not happen when only one mfakto-instance was running.
- There's been reports of mfakto-crashes when other GPU-bound tools or GPU-Z
  were running.

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

Read mfakto.ini and think before edit. ;)



#########
# 6 FAQ #
#########

Q Does mfakto support multiple GPUs?
A Not tested yet, but using the commandline option "-d <GPU number>" you should
  be able to specify which GPU to use for each specific mfakto instance.
  Please read the next question, too.

Q Can I run multiple instances of mfakto on the same computer?
A Well, normally yes, but there is a bug that sometimes freezes the whole
  computer when one of multiple instances exits.

Q Which tasks should I assign to mfakto?
A Currently, the 71-bit kernel is the only fast one (though everything up to
  95-bit will work). Due to an internal optimization, factors > 2^64 are 5-10%
  faster than factors up to 2^64. So best is to assign TF work between 2^64 and
  2^71 for now. Other fast kernels will follow.

###########
# 7 .plan #
###########

0.06
- add a barrett kernel
- SievePrimesAdjust

not planned for a specific release / ongoing
- keep features in sync with mfaktc
- performance improvements whenever I find them ;)
- find the reason for the occasional aborts
- documentation and comments in code
- find a smarter way for the vectors (not so many kernels and functions, but a
  compile-time definition)
- combine barrett and vectors
- full 95-bit implementation
- once stable, enable uploading results to primenet

