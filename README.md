# smith-waterman-optimization
_Created as a part of the CSC586C (Data Management on Modern Computer Architectures) course at University of Victoria under supervision of Sean Chester._

Contributors: Ze Shi Li, Rohith Pudari, Haris Smajlovic.

Repository contains performance optimizations for Linear gap Smith-Waterman algorithm.

## Optimizations
So far we have a `baseline`, `bithacked`, `bithacked-striped`, `windowed`, `simd-alpern-256` and `simd-alpern-512` version of the very same algorithm.
- Baseline: A straight forward baseline version of the SW algorithm.
- Bithacked: Baseline version with heavy branching replaced with bithacks.
- Bithacked-striped: Bithacked version with an access pattern that is more L1 cache friendly.
- Windowed: A version of hypothetical scenario in which dynamic programming matrix is not needed and only the maximum value in the matrix is searched for.
- 256-wide SIMDed (Alpern technique): A SIMDed baseline using 256bit long SIMD registers and technique from [Alpern et al](https://dl.acm.org/doi/10.1145/224170.224222).
- 512-wide SIMDed (Alpern technique): A SIMDed baseline using 512bit long SIMD registers and technique from [Alpern et al](https://dl.acm.org/doi/10.1145/224170.224222). **Note: Check if your CPU supports AVX-512 first. Otherwise the benchmark will still run, but with no valid results.**

## Testing
In order to benchmark the algorithm use `perf` (for now -- sorry non-linux users). So just compile `benchmark.cpp` and then run `perf` on the executable. Don't forget to provide version string as a CLI argument.

### Examples
- For baseline set `version=base` in your bash
- For bithacked set `version=bithacked` in your bash
- For bithacked-striped set `version=bithacked-striped` in your bash
- For windowed set `version=windowed` in your bash
- For 256-wide SIMDed (Alpern technique) set `version=simd-alpern-256` in your bash
- For 512-wide SIMDed (Alpern technique) set `version=simd-alpern-512` in your bash

and then do
```bash
exe_path=benchmark_${version}.out && g++ -march=native -Wall -Og -std=c++17 -o $exe_path benchmark.cpp && perf stat -e L1-dcache-load-misses:u,LLC-load-misses:u,cache-misses:u,cache-references:u,branch-misses:u,page-faults:u,cycles:u,L1-dcache-stores:u,instructions:u ./$exe_path $version
```
