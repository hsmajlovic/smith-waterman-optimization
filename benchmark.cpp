/**
 * A small application to benchmark the performance of hand-built sort algorithms.
 */

#include <iostream>  // std::cout
#include <string>
#include <utility>
#include "benchmarking/timing.hpp"
#include "benchmarking/data-generation.hpp"
#include "assert.h"
#include "smith_waterman.cpp"

struct compare_pairs
{
	template < typename T >
		T operator () ( std::pair< T, T > data ) const
		{
			std::cout << "New pair: " << std::endl;
            std::cout << data.first << " " << data.second<< std::endl;
			
			smith_waterman(data);
            return "0";
		}
};


int main()
{
	auto num_pairs  = 1u;
	auto string_len = 1u << 3;

    // For random numbers, one must first seed the random number generator. This is the idiomatic
    // approach for the random number generator libraries that we have chosen.
    std::srand ( static_cast< uint32_t >( std::time(0) ) );
	auto const test_cases = csc586::benchmark::uniform_rand_vec_of_vec< std::string >( num_pairs, string_len );
    // std::cout<< test_cases[0].first << test_cases[0].second << "\n";
	auto const run_time   = csc586::benchmark::benchmark( compare_pairs{}
                                                          , test_cases );

    std::cout << "Average time (us): " << run_time << std::endl;

	return 0;
}
