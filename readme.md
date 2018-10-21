
# Matching Benchmarks
I wanted to benchmark different strategies for matching two sets of points, in this case jsut by distance. The classes' names include 'grid' because I was going to benchmark ways of assinging points to grid cells first.
Instead I started with the naive O(nn) matching. I think for moderate sizes this should be faster on the GPU, but the grid is probably faster for all but tiny sizes on the CPU.

So far I have:
  - CpuNaive: simple O(nn) matching
  - CpuNaiveMultithreaded: slice first point set into N groups and assign N threads the work. Actually I use 'tasks' in the C++11 terminology.
  - CudaNaive: The file contains 3 different kernels and 3 different approaches.
      1. First is each item in first point set gets its own thread and has a `N` loop over each second item. This is simple in that there is no contention amongst threads if it's okay for the matching to be injective.
      2. Each pair gets its own thread. Faster as `N` grows. Not always correct, if there is contention it could be wrong. Injectivity still not handled.
      3. Each pair gets its own thread, each A point has it's own `distThresh`, which is atomically read/written and ensures matching distance is always decreasing. Initializes `distThresh` to a default as other methods do.
      4. (*TODO*) Method (3), plus another array to prevent cross-matching and handle injectivity.


### Result
```
slee/stuff/grid_benchmarks/build git/master*  
❯ ./grid_bench

 ============================================== 
        For data size 512
 -- cpu took 0.28ms for one step.
 -- cpu took 0.26ms for one step.
 -- cpu took 0.21ms for one step.
 -- cpu_mt took 0.51ms for one step.
 -- cpu_mt took 0.32ms for one step.
 -- cpu_mt took 1.23ms for one step.
 -- cuda took 99.59ms for one step.
 -- cuda took 0.09ms for one step.
 -- cuda took 0.09ms for one step.

 ============================================== 
        For data size 1024
 -- cpu took 0.45ms for one step.
 -- cpu took 0.43ms for one step.
 -- cpu took 0.43ms for one step.
 -- cpu_mt took 0.45ms for one step.
 -- cpu_mt took 0.33ms for one step.
 -- cpu_mt took 0.41ms for one step.
 -- cuda took 0.35ms for one step.
 -- cuda took 0.29ms for one step.
 -- cuda took 0.29ms for one step.

 ============================================== 
        For data size 2048
 -- cpu took 1.64ms for one step.
 -- cpu took 1.72ms for one step.
 -- cpu took 1.89ms for one step.
 -- cpu_mt took 0.88ms for one step.
 -- cpu_mt took 1.23ms for one step.
 -- cpu_mt took 1.87ms for one step.
 -- cuda took 0.67ms for one step.
 -- cuda took 0.28ms for one step.
 -- cuda took 0.29ms for one step.

 ============================================== 
        For data size 4096
 -- cpu took 7.49ms for one step.
 -- cpu took 7.11ms for one step.
 -- cpu took 7.04ms for one step.
 -- cpu_mt took 3.49ms for one step.
 -- cpu_mt took 3.66ms for one step.
 -- cpu_mt took 3.32ms for one step.
 -- cuda took 1.36ms for one step.
 -- cuda took 1.17ms for one step.
 -- cuda took 1.16ms for one step.

 ============================================== 
        For data size 8192
 -- cpu took 30.08ms for one step.
 -- cpu took 27.55ms for one step.
 -- cpu took 27.55ms for one step.
 -- cpu_mt took 12.75ms for one step.
 -- cpu_mt took 14.82ms for one step.
 -- cpu_mt took 11.59ms for one step.
 -- cuda took 5.45ms for one step.
 -- cuda took 3.93ms for one step.
 -- cuda took 3.91ms for one step.

 ============================================== 
        For data size 16384
 -- cpu took 110.46ms for one step.
 -- cpu took 110.73ms for one step.
 -- cpu took 108.57ms for one step.
 -- cpu_mt took 48.39ms for one step.
 -- cpu_mt took 46.06ms for one step.
 -- cpu_mt took 41.01ms for one step.
 -- cuda took 18.21ms for one step.
 -- cuda took 19.29ms for one step.
 -- cuda took 17.35ms for one step.

 ============================================== 
        For data size 32768
 -- cpu took 437.07ms for one step.
 -- cpu took 439.73ms for one step.
 -- cpu took 437.16ms for one step.
 -- cpu_mt took 191.97ms for one step.
 -- cpu_mt took 166.68ms for one step.
 -- cpu_mt took 155.39ms for one step.
 -- cuda took 69.88ms for one step.
 -- cuda took 68.12ms for one step.
 -- cuda took 68.60ms for one step.

 ============================================== 
        For data size 65536
 -- cpu took 1750.24ms for one step.
 -- cpu took 1738.18ms for one step.
 -- cpu took 1760.50ms for one step.
 -- cpu_mt took 724.07ms for one step.
 -- cpu_mt took 768.91ms for one step.
 -- cpu_mt took 693.79ms for one step.
 -- cuda took 240.20ms for one step.
 -- cuda took 233.93ms for one step.
 -- cuda took 234.66ms for one step.
```
