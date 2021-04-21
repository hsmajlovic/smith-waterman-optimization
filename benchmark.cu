#ifndef CUDA_BLOCK_SIZE
	#define CUDA_BLOCK_SIZE (1 << BLOCK_SIZE_SCALE)
#endif
#ifndef QUANTITY
	#define QUANTITY (1 << QUANTITY_SCALE)
#endif
#ifndef SIZE
	#define SIZE (1 << SIZE_SCALE)
#endif
#ifndef WINDOW_SIZE
	#define WINDOW_SIZE (1 << WINDOW_SIZE_SCALE)
#endif

// for cuda error checking
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            return; \
        } \
    } while (0)


#include <iostream>  // std::cout
#include <string>
#include <utility>
#include <set>
#include "benchmarking/timing.hpp"
#include "benchmarking/data-generation.hpp"
#include "assert.h"
#include "sw_cuda_alpern.cu"
#include "sw_cuda_windowed.cu"


int main(int argc, char** argv)
{
	std::string version(argv[argc - 1]);
	std::vector<std::string> versions_list = { "cuda-alpern", "cuda-windowed" };
	std::set<std::string> versions (versions_list.begin(), versions_list.end());
	const bool is_in = versions.find(version) != versions.end();
	if (!is_in) std::cout << "Incorrect version provided: " << version << std::endl;
	assert (is_in);

    // For random numbers, one must first seed the random number generator. This is the idiomatic
    // approach for the random number generator libraries that we have chosen.
    std::srand ( static_cast< uint32_t >( std::time(0) ) );
	auto const test_cases = csc586::benchmark::uniform_rand_vec_of_vec< std::string >( QUANTITY, SIZE );

	if (version == "cuda-alpern")
		sw_cuda_alpern(test_cases);
	else if (version == "cuda-windowed")
		sw_cuda_windowed(test_cases);

	return 0;
}