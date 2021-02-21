/**
 * A small application to benchmark the performance of hand-built sort algorithms.
 */

#include <iostream>  // std::cout
#include <string>
#include <utility>
#include "benchmarking/timing.hpp"
#include "benchmarking/data-generation.hpp"
#include "assert.h"
#include "smith_waterman_base.cpp"
#include "smith_waterman_windowed.cpp"
#include "smith_waterman_striped.cpp"
// #include "gnuplot-iostream.h"


struct base_sw
{
	template < typename T >
		T operator () ( std::pair< T, T > data ) const
		{
			smith_waterman_base(data);
            return "0";
		}
};


struct windowed_sw
{
	template < typename T >
		T operator () ( std::pair< T, T > data ) const
		{
			smith_waterman_windowed(data);
            return "0";
		}
};


struct striped_sw
{
	template < typename T >
		T operator () ( std::pair< T, T > data ) const
		{
			smith_waterman_striped(data);
            return "0";
		}
};


int main(int argc, char** argv)
{
	auto num_pairs  = 1u;
	auto string_len = 1u << 15;
	std::string version(argv[argc - 1]);

    // For random numbers, one must first seed the random number generator. This is the idiomatic
    // approach for the random number generator libraries that we have chosen.
    std::srand ( static_cast< uint32_t >( std::time(0) ) );
	auto const test_cases = csc586::benchmark::uniform_rand_vec_of_vec< std::string >( num_pairs, string_len );
	auto const run_time   = version == "striped" ? csc586::benchmark::benchmark(striped_sw{}, test_cases) :
							version == "windowed" ? csc586::benchmark::benchmark(windowed_sw{}, test_cases) :
							csc586::benchmark::benchmark(base_sw{}, test_cases);

    std::cout << "Average time (us): " << run_time << std::endl;

	return 0;
}
