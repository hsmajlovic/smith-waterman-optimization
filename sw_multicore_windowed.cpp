/* 
NOTE: The code in here is just a POC of the hypothetical (and impossible) scenario in which
the lowest level caches are utilized the most.
Only the bottleneck of the algorithm is benchmarked, i.e. the population of the matrix.
Idea is to have a window of size that can surely fit in L1 cache (L1 cache size for our machine is 32KB)
and then traverse that window over the input strings in order to find the max element of the matrix.
That way we can ge trid of the matrix completely and keep track only of the window (of size w*w)
and some auxiliary data of size approx. 2*w. In our case we hardcoded w to be 256 elements long.
Each element is a 4B int. 
Please note that even the correctness of the algorithm is sacrifised in order to maximize cache utilization as
the side_leftover and top_leftover are mocked with incorrect values.
*/
#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include <omp.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include "traceback.cpp"


template < typename T >
    void sw_multicore_windowed(std::vector<std::pair< T, T >> const sequences){

        // TODO switch from std::pair to using std::vector
        unsigned int const quantity = sequences.size();
 
        // Threads number
        const int num_threads = omp_get_max_threads();

        // Setup sizes
        unsigned int size = sequences[0].first.size();
        unsigned int window_size = 1u << 8;  // Experimental best: 1u << 8, still 1u << 8 for some reason

        // Instantiate matrices for reduction
        std::vector<std::vector<std::vector<int>>> windows(
            num_threads, std::vector<std::vector<int>>( window_size + 1, std::vector<int>( window_size + 1 )));
        
        #pragma omp parallel for
        for (unsigned int s =0; s< quantity; ++s) {
            const int t = omp_get_thread_num();

            // instantiate a matrix 
            T s1 = sequences[s].first;
            T s2 = sequences[s].second;
            unsigned int size = s1.size();

            unsigned int batches_no = size / window_size;
            std::vector<int> top_leftover(window_size, 0);
            std::vector<int> side_leftover(window_size, 0);

            int gaps(-2), mismatch(-2), match(3), max_element(0);
            unsigned int max_element_i(0), max_element_j(0);
            int diagonal_value, top_value, left_value;
            
            for (unsigned int bi = 0; bi < batches_no; ++bi) {
                unsigned int offset_i = bi * window_size;
                T s1_piece = s1.substr(offset_i, offset_i + window_size);
                for (unsigned int bj = 0; bj < batches_no; ++bj) {
                    unsigned int offset_j = bj * window_size;
                    T s2_piece = s2.substr(offset_j, offset_j + window_size);
                    
                    for (unsigned int i = 0; i < window_size; ++i) {
                        windows[t][0][i] = top_leftover[i];
                        windows[t][i][0] = side_leftover[i];
                    }
                    
                    for (unsigned int i = 1; i < window_size; ++i) {
                        for (unsigned int j = 1; j < window_size; ++j) {
                            diagonal_value = windows[t][i-1][j-1] + (s1_piece[i - 1] == s2_piece[j - 1] ? match : mismatch);
                            top_value = windows[t][i-1][j] + gaps;
                            left_value = windows[t][i][j-1] + gaps;
                            int temp = top_value - ((top_value - left_value) & ((top_value - left_value) >> (sizeof(int) * 8 - 1)));
                            int target_value = diagonal_value - ((diagonal_value - temp) & ((diagonal_value - temp) >> (sizeof(int) * 8 - 1)));  // std::max(diagonal_value, std::max(top_value, left_value));
                            windows[t][i][j] = target_value - (target_value & (target_value >> (sizeof(int) * 8 - 1)));  // (target_value > 0) ? target_value : 0;
                            if (target_value > max_element) {
                                max_element = target_value;
                                max_element_i = offset_i + i;
                                max_element_j = offset_j + j;
                            }
                        }
                    }

                    for (unsigned int i = 0; i < window_size; ++i) {
                        top_leftover[i] = windows[t][window_size - 1][i];
                        side_leftover[i] = windows[t][i][window_size - 1];
                    }
                }
            }
            //std::cout << max_element << " " << max_element_i << " " << max_element_j << std::endl;
        }

        
}