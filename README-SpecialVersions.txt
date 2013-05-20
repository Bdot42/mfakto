The SpecialVersions_x64.zip and x32.zip packages contain mfakto.exe files
compiled for special purposes:

mfakto-64k :  A binary optimized for 64k L1 cache. This is useful for most AMD
              CPUs, but also for Intel CPUs when going to high SievePrimes
              values (>100k).
mfakto-var :  Compiled to allow SieveSize configuration via mfakto.ini. Most
              useful for testing which SieveSize is optimal, but 1-3% slower
              sieving than the fixed-size compiled files (for the same SieveSize).
mfakto-pi  :  For each block, the OpenCL performance counters are read and
              displayed. This allows for accurate measuring of transfer and
              kernel execution speed. Intended for performance tests.

Apart from these settings, these binaries are the same as the corresponding 
normal package, which is optimized for 32k L1 cache (most Intel CPUs).

