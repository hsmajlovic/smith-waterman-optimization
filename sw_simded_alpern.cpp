#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include "nmmintrin.h" // for SSE4.2
#include "immintrin.h" // for AVX


template < typename T >
    void smith_waterman_simded_alpern(std::vector<std::pair< T, T >> sequences){
        // instantiate a matrix 
        unsigned int size     = sequences[0].first.size();
        unsigned int quantity = sequences.size();
        std::vector<std::vector<__m256i>> matrix(size + 1, std::vector<__m256i>(size + 1, _mm256_setzero_si256()));

        // Instantiate SIMDed scores
        const __m256i gap      = _mm256_set1_epi32(-2);
        const __m256i mismatch =  _mm256_set1_epi32(-2);
        const __m256i match    =  _mm256_set1_epi32(3);
        
        // SIMD size
        unsigned int sse_s     = 8;

        for (unsigned int k = 0; k < quantity; k += sse_s) {
            // Target SIMDed values
            __m256i max_element   = _mm256_setzero_si256();
            // __m256i max_element_i = _mm256_setzero_si256();
            // __m256i max_element_j = _mm256_setzero_si256();
            
            // Auxiliary values
            __m256i diagonal_value, top_value, left_value,
                    top_left, top_left_shifted, temp, diagonal_temp,
                    diagonal_temp_shifted, target_value, target_value_shifted,
                    top_left_and, diagonal_temp_and, target_value_and,
                    max_target_shifted, max_target;

            // Construct next SIMD sized batch of chars
            std::vector<char*> i_seq(size, new char[sse_s]);
            std::vector<char*> j_seq(size, new char[sse_s]);
            // Construct next SIMD sized batch of chars
            for (unsigned int p = 0; p < sse_s; ++p) {
                for (unsigned int i = 0; i < size; ++i) {
                    i_seq[i][p] = sequences[k + p].first[i];
                    j_seq[i][p] = sequences[k + p].second[i];
                }
            }
            
            for (unsigned int i = 1; i < size + 1; ++i){
                for(unsigned int j = 1; j < size + 1; ++j) {
                    // match_val ~ data.first[i - 1] == data.second[j - 1] ? match : mismatch)
                    // long int mask = atoi(i_seq[i]) ^ atoi(j_seq[j]);
                    const int mask = 19;
                    __m256i match_val = _mm256_blend_epi32(mismatch, match, mask);  // (sequences[k].first[i - 1] == sequences[k].second[j - 1] ? match : mismatch);
                    
                    // diagonal_value ~ matrix[i-1][j-1] + match_val
                    diagonal_value   = _mm256_add_epi32 ( matrix[i-1][j - 1], match_val );
                    // top_value ~ matrix[i-1][j] + gap
                    top_value        = _mm256_add_epi32 ( matrix[i-1][j], gap);
                    // left_value ~ matrix[i][j-1] + gap
                    left_value       = _mm256_add_epi32 ( matrix[i][j - 1], gap) ;
                    
                    // Calculate target_value ~ std::max(diagonal_value, std::max(top_value, left_value))
                    top_left         = _mm256_sub_epi32 ( top_value, left_value );
                    top_left_shifted = _mm256_srli_epi32( top_left,  31 );
                    top_left_and     = top_left & top_left_shifted;
                    temp             = _mm256_sub_epi32 ( top_value, top_left_and );
                    diagonal_temp    = _mm256_sub_epi32 ( diagonal_value, temp );
                    diagonal_temp_and = diagonal_temp & diagonal_temp_shifted;
                    diagonal_temp_shifted  = _mm256_srli_epi32( diagonal_temp, 31 );
                    target_value           = _mm256_sub_epi32 ( diagonal_value, diagonal_temp_and );

                    // Calculate  matrix[i][j] ~ (target_value > 0) ? target_value : 0
                    target_value_shifted   = _mm256_srli_epi32( target_value, 31 );
                    target_value_and       = target_value & target_value_shifted;
                    matrix[i][j]           = _mm256_sub_epi32 ( target_value, target_value_and );

                    // Update max_element and coordinates if the target_value is larger
                    max_target         = _mm256_sub_epi32(target_value, max_element);
                    max_target_shifted = _mm256_srli_epi32(max_target, 31);
                    max_element        = _mm256_sub_epi32(target_value, (max_target & max_target_shifted));
                    
                    // if (target_value > max_element) {
                    //     max_element = target_value;
                    //     max_element_i = i;
                    //     max_element_j = j;
                    // }
                }
            }
            if (k % (1u << 9) == 0) {
                auto const vec = reinterpret_cast< int const * >( &max_element );
                std::cout << vec[0] << std::endl;
            }
        
            // // traceback

            // std::vector<std::pair<int, int>> traceback_indices;
            // std::string alignment_str_1("");
            // std::string alignment_str_2("");
            // int current_i = max_element_i;
            // int current_j = max_element_j;
            // while (matrix[current_i][current_j]) {
            //     traceback_indices.push_back(std::make_pair(current_i, current_j));
            //     diagonal_value = matrix[current_i][current_j] - (s1[current_i - 1] == s2[current_j - 1] ? match : mismatch);
            //     top_value = matrix[current_i][current_j] - gaps;
            //     left_value = matrix[current_i][current_j] - gaps;
            //     if (diagonal_value == matrix[current_i-1][current_j-1]) {
            //         current_i = current_i - 1;
            //         current_j = current_j - 1;
            //         alignment_str_1 += s1[current_i];
            //         alignment_str_2 += s2[current_j];
            //     } else if (top_value == matrix[current_i-1][current_j]) {
            //         current_i = current_i - 1;
            //         alignment_str_1 += s1[current_i];
            //         alignment_str_2 += '-';
            //     } else {
            //         current_j = current_j - 1;
            //         alignment_str_1 += '-';
            //         alignment_str_2 += s2[current_j];
            //     }
            // }

            // std::reverse(alignment_str_1.begin(), alignment_str_1.end());
            // std::reverse(alignment_str_2.begin(), alignment_str_2.end());

            // std::cout << alignment_str_1 << " " << alignment_str_2 << std::endl;

        }

}
