#ifndef SSE_S
	#ifdef __AVX512F__
	#define SSE_S 16
	#elif defined __AVX2__
	#define SSE_S 8
	#elif defined __SSE2__ && defined __SSE4_1__
	#define SSE_S 4
	#else
	#define SSE_S 1
	#endif
#endif


#include <iostream>  // std::cout
#include <string>
#include <utility>
#include <set>
#include "benchmarking/timing.hpp"
#include "benchmarking/data-generation.hpp"
#include "assert.h"
#include "sw_base.cpp"
#include "sw_windowed.cpp"
#include "sw_parallel_windowed.cpp"
#include "sw_bithacked.cpp"
#include "sw_bithacked_striped.cpp"
#include "sw_simded_alpern.cpp"
#include "sw_multicore_alpern.cpp"


struct base_sw
{
	template < typename T >
		T operator () ( std::pair< T, T > data ) const
		{
			sw_base(data);
            return "0";
		}
};


struct windowed_sw
{
	template < typename T >
		T operator () ( std::vector<std::pair< T, T >> data  ) const
		{
			sw_windowed(data);
            return "0";
		}
};

struct parallel_windowed_sw
{
	template < typename T >
		T operator () ( std::vector<std::pair< T, T >> data  ) const
		{
			sw_parallel_windowed(data);
            return "0";
		}
};


struct bithacked_sw
{
	template < typename T >
		T operator () ( std::pair< T, T > data ) const
		{
			sw_bithacked(data);
            return "0";
		}
};


struct bithacked_striped_sw
{
	template < typename T >
		T operator () ( std::vector<std::pair< T, T >>  data ) const
		{
			sw_bithacked_striped(data);
            return "0";
		}
};


struct simded_alpern_sw
{
	template < typename T >
		T operator () ( std::vector<std::pair< T, T >> data ) const
		{
			sw_simded_alpern(data);
            return "0";
		}
};


struct multicore_alpern_sw
{
	template < typename T >
		T operator () ( std::vector<std::pair< T, T >> data ) const
		{
			sw_multicore_alpern(data);
            return "0";
		}
};


int main(int argc, char** argv)
{
	auto num_pairs  = 1u << 13;
	auto string_len = 1u << 10;
	omp_set_num_threads( THRD_CNT );
	
	std::string version(argv[argc - 1]);
	std::vector<std::string> versions_list = { 
		"base", "windowed", "parallel-windowed", "bithacked", "bithacked-striped", "simd-alpern", "multicore-alpern"};
	std::set<std::string> versions (versions_list.begin(), versions_list.end());
	const bool is_in = versions.find(version) != versions.end();
	if (!is_in) std::cout << "Incorrect version provided: " << version << std::endl;
	assert (is_in);

    // For random numbers, one must first seed the random number generator. This is the idiomatic
    // approach for the random number generator libraries that we have chosen.
    std::srand ( static_cast< uint32_t >( std::time(0) ) );
	auto const test_cases = csc586::benchmark::uniform_rand_vec_of_vec< std::string >( num_pairs, string_len );
	auto const run_time   = version == "multicore-alpern" ? csc586::benchmark::benchmark_once(multicore_alpern_sw{}, test_cases) :
							version == "simd-alpern" ? csc586::benchmark::benchmark_once(simded_alpern_sw{}, test_cases) :
							version == "parallel-windowed" ? csc586::benchmark::benchmark_once(parallel_windowed_sw{}, test_cases) :
							version == "windowed" ? csc586::benchmark::benchmark_once(windowed_sw{}, test_cases) :
							version == "bithacked" ? csc586::benchmark::benchmark(bithacked_sw{}, test_cases) :
							version == "bithacked-striped" ? csc586::benchmark::benchmark_once(bithacked_striped_sw{}, test_cases) :
							csc586::benchmark::benchmark(base_sw{}, test_cases);

    std::cout << "Average time (us): " << run_time << std::endl;

	return 0;
}
