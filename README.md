# Smith-Waterman performance optimizations
_Created as a part of the CSC586C (Data Management on Modern Computer Architectures) course at University of Victoria under supervision of Sean Chester._

Contributors: Ze Shi Li, Rohith Pudari, Haris Smajlovic.

Repository contains performance optimizations for [Linear gap Smith-Waterman algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm#Linear).

## Optimizations

**Note: Check if your CPU supports SSE2/4, AVX2 and/or AVX-512 first. Otherwise the SIMD benchmark will still run, but with no valid results.**

So far we have a `baseline`, `bithacked`, `bithacked-striped`, `multicore-windowed`, `windowed`, `simd-alpern` and `multicore-alpern` version of the very same algorithm for a CPU, and `cuda-alpern` for a GPU:
- Baseline: A straight forward baseline version of the SW algorithm.
- Bithacked: Baseline version with heavy branching replaced with bithacks.
- Bithacked-striped: Bithacked version with an access pattern that is more L1 cache friendly.
- Windowed: A version of hypothetical scenario in which dynamic programming matrix is not needed and only the maximum value in the matrix is searched for.
- Multicore-windowed: Using the technique above, spreads across multiple CPU cores.
- SIMDed (Alpern technique): A SIMDed baseline using widest registers that your CPU supports and inter-alignment technique from [Alpern et al](https://dl.acm.org/doi/10.1145/224170.224222).
- Multicore (Alpern technique): Just a SIMDed technique above spread accross multiple CPU cores.
- CUDA (Alpern technique): A SIMTed baseline using inter-alignment technique akin to SIMD Alpern technique above.

## Testing
In order to benchmark the CPU solutions use `perf` (for now -- sorry non-linux users). So just compile `benchmark.cpp` and then run `perf` on the executable.

For testing the GPU solutions just compile and run `benchmark.cu`.

Don't forget to provide version string as a CLI argument.

### CPU Examples
- For baseline set `version=base` in your bash
- For bithacked set `version=bithacked` in your bash
- For bithacked-striped set `version=bithacked-striped` in your bash
- For windowed set `version=windowed` in your bash
- For multicore-windowed set `version=multicore-windowed` in your bash
- For SIMDed (Alpern technique) set `version=simd-alpern` in your bash
- For multicore set `version=multicore-alpern` in your bash

and then do
```bash
exe_path=benchmark_${version}.out && \
g++ -D THRD_CNT=2 -march=native -fopenmp -Wall -Og -std=c++17 -o $exe_path benchmark.cpp && \
perf stat -e cycles:u,instructions:u ./$exe_path $version
```

### GPU Examples
- For CUDA (Aplern technique) set `version=cuda-alpern` in your bash

and then do
```bash
exe_path=benchmark_${version}.out && \
nvcc -O3 -D QUANTITY_SCALE=13 -D SIZE_SCALE=10 -D BLOCK_SIZE_SCALE=7 -o $exe_path benchmark.cu && \
./$exe_path $version
```

### Results

| Version        | insn per cycle  | Seconds  |
| ------------- |:-------------:| -----:|
| Bithacked-striped (optimised-base) | 2.77 | 41.50 |
| SIMD-alpern | 2.95      |   3.76 |
| multicore-alpern | 2.26  | 2.45 |
