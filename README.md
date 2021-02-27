# smith-waterman-optimization
_Created as a part of the CSC586C (Data Management on Modern Computer Architectures) course at University of Victoria under supervision of Sean Chester._

Contributors: Ze Shi Li, Rohith Pudari, Haris Smajlovic.

Repository contains performance optimizations for Linear gap Smith-Waterman algorithm.

## Optimizations
So far we have a `baseline`, `bithacked`, `bithacked-striped` and `windowed` version of the very same algorithm.
- Baseline: A straight forward baseline version of the SW algorithm.
- Bithacked: Baseline version with heavy branching replaced with bithacks.
- Bithacked-striped: Bithacked version with an access pattern that is more L1 cache friendly.
- Windowed: A version of hypothetical scenario in which dynamic programming matrix is not needed and only the maximum value in the matrix is searched for.

## Testing
In order to benchmark the algorithm use `perf` (for now -- sorry non-linux users). So just compile `benchmark.cpp` and then run `perf` on the executable. Don't forget to provide version string as a CLI argument.

### Examples
- For baseline set `version=base` in your bash
- For bithacked set `version=bithacked` in your bash
- For bithacked-striped set `version=bithacked-striped` in your bash
- For windowed set `version=windowed` in your bash

and then do
```bash
exe_path=benchmark_${version}.out && g++ -Wall -Og -std=c++17 -o $exe_path benchmark.cpp && perf stat -e L1-dcache-load-misses:u,LLC-load-misses:u,cache-misses:u,cache-references:u,branch-misses:u,page-faults:u,cycles:u,L1-dcache-stores:u,instructions:u ./$exe_path $version
```
