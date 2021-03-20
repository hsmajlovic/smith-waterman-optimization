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
#include "immintrin.h" // for AVX
#include "traceback.cpp"

#ifdef __AVX2__
template < typename T >
    void sw_parallel_windowed_256(std::vector<std::pair< T, T >> const sequences, unsigned int const quantity, unsigned int const size){
        // Threads number
        const int num_threads = omp_get_max_threads();

        // Setup sizes
        unsigned int window_size = 1u << 8;  // Experimental best: 1u << 8, still 1u << 8 for some reason

        // Instantiate matrices for reduction
        std::vector<std::vector<std::vector<__m256i>>> windows(
            num_threads, std::vector<std::vector<__m256i>>( window_size + 1, std::vector<__m256i>( window_size + 1 )));
        
        // Instantiate SIMDed scores
        const __m256i gap      = _mm256_set1_epi32( -2 );
        const __m256i mismatch = _mm256_set1_epi32( -2 );
        const __m256i match    = _mm256_set1_epi32( 3 );
        const __m256i zeros    = _mm256_setzero_si256();
        
        #pragma omp parallel for
        for (unsigned int s = 0; s < quantity; s += SSE_S) {
            const int t = omp_get_thread_num();

            // Target SIMDed values
            __m256i max_element;
            __m256i max_element_i;
            __m256i max_element_j;

            // Auxiliary values
            __m256i diagonal_value;
            __m256i top_value;
            __m256i left_value;
            __m256i temp_value;
            __m256i target_value;
            __m256i i_vectorized;
            __m256i j_vectorized;
            __m256i max_element_updated;
            __m256i mask;
            __m256i match_val;

            // Char batching containers
            std::vector<__m256i> i_seq( size );
            std::vector<__m256i> j_seq( size );
            int char_batch_i[ SSE_S ];
            int char_batch_j[ SSE_S ];
            std::vector<__m256i> s1_piece;
            std::vector<__m256i> s2_piece;

            // Set target values
            max_element   = _mm256_setzero_si256();
            max_element_i = _mm256_setzero_si256();
            max_element_j = _mm256_setzero_si256();

            // Construct next SIMDed batch of chars
            for (unsigned int i = 0; i < size; ++i) {
                for (unsigned int p = 0; p < SSE_S; ++p) {
                    char_batch_i[p] = int(sequences[s + p].first[i]);
                    char_batch_j[p] = int(sequences[s + p].second[i]);
                }
                i_seq[i] = _mm256_load_si256( ( __m256i * ) char_batch_i );
                j_seq[i] = _mm256_load_si256( ( __m256i * ) char_batch_j );
            }

            unsigned int batches_no = size / window_size;
            std::vector<__m256i> top_leftover( window_size );
            std::vector<__m256i> side_leftover( window_size );

            for ( unsigned int bi = 0; bi < batches_no; ++bi ) {
                unsigned int offset_i = bi * window_size;
                s1_piece = std::vector<__m256i>( i_seq.begin() + offset_i, i_seq.begin() + offset_i + window_size );
                for ( unsigned int bj = 0; bj < batches_no; ++bj ) {
                    unsigned int offset_j = bj * window_size;
                    s2_piece = std::vector<__m256i>( j_seq.begin() + offset_j, j_seq.begin() + offset_j + window_size );
                    
                    for (unsigned int i = 0; i < window_size; ++i) {
                        windows[t][0][i] = top_leftover[i];
                        windows[t][i][0] = side_leftover[i];
                    }
                    
                    for (unsigned int i = 1; i < window_size; ++i) {
                        i_vectorized = _mm256_set1_epi32( i );
                        for (unsigned int j = 1; j < window_size; ++j) {
                            j_vectorized = _mm256_set1_epi32( j );

                            // match_val ~ data.first[i - 1] == data.second[j - 1] ? match : mismatch)
                            mask      = _mm256_cmpeq_epi32( s1_piece[i-1], s2_piece[j-1] );
                            match_val = _mm256_blendv_epi8( mismatch, match, mask );
                            
                            // diagonal_value ~ matrix[i-1][j-1] + match_val
                            diagonal_value   = _mm256_add_epi32 ( windows[t][i-1][j-1], match_val );
                            // top_value ~ matrix[i-1][j] + gap
                            top_value        = _mm256_add_epi32 ( windows[t][i-1][j], gap );
                            // left_value ~ matrix[i][j-1] + gap
                            left_value       = _mm256_add_epi32 ( windows[t][i][j-1], gap);
                            
                            // Calculate target_value ~ std::max(diagonal_value, std::max(top_value, left_value))
                            temp_value    = _mm256_max_epi32( top_value, left_value );
                            target_value  = _mm256_max_epi32( diagonal_value, temp_value );
                            // Calculate  matrix[i][j] ~ (target_value > 0) ? target_value : 0
                            windows[t][i][j]  = _mm256_max_epi32( target_value, zeros );
                            // Update max_element and coordinates if the target_value is larger
                            max_element         = _mm256_max_epi32( max_element, target_value );
                            max_element_updated = _mm256_cmpeq_epi32( max_element, target_value );
                            max_element_i       = _mm256_blendv_epi8( max_element_i, i_vectorized, max_element_updated );
                            max_element_j       = _mm256_blendv_epi8( max_element_j, j_vectorized, max_element_updated );

                        }
                    }

                    for (unsigned int i = 0; i < window_size; ++i) {
                        top_leftover[i] = windows[t][window_size - 1][i];
                        side_leftover[i] = windows[t][i][window_size - 1];
                    }
                }
            }
            if (s % (1u << 9) == 0) {
                auto const vec = reinterpret_cast< int const * >( &max_element );
                std::cout << vec[0] << std::endl;
            }
        }
    }
#endif

template < typename T >
    void sw_parallel_windowed(std::vector<std::pair< T, T >> const sequences){
        unsigned int const size     = sequences[0].first.size();
        unsigned int const quantity = sequences.size();

        #ifdef __AVX2__
        sw_parallel_windowed_256(sequences, quantity, size);
        #else
        std::cout << "Parallel windowed works only with AVX2." << std::endl;
        #endif
    }
