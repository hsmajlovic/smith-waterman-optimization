#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include "nmmintrin.h" // for SSE4.2
#include "immintrin.h" // for AVX


template < typename T >
    void sw_simded_alpern(std::vector<std::pair< T, T >> sequences){
        // TODO switch from std::pair to using std::vector
        // instantiate a matrix 
        unsigned int size     = sequences[0].first.size();
        unsigned int quantity = sequences.size();

        #ifdef __AVX512F__
        std::cout << "Using 512 bits wide registers ... " << std::endl;
        std::vector<std::vector<__m512i>> matrix(size + 1, std::vector<__m512i>(size + 1));

        // Instantiate SIMDed scores
        const __m512i gap      = _mm512_set1_epi32(-2);
        const __m512i mismatch =  _mm512_set1_epi32(-2);
        const __m512i match    =  _mm512_set1_epi32(3);
        const __m512i zeros    = _mm512_setzero_si512();

        // Target SIMDed values
        __m512i max_element, max_element_i, max_element_j;

        // Auxiliary values
        __m512i diagonal_value, top_value, left_value, temp_value,
                target_value, i_vectorized, j_vectorized, match_val;
        __mmask16 mask, max_element_updated;
        
        // SIMD size
        unsigned int sse_s     = 16;

        // Char batching containers
        std::vector<__m512i> i_seq( size );
        std::vector<__m512i> j_seq( size );
        int char_batch_i[ sse_s ];
        int char_batch_j[ sse_s ];

        for (unsigned int k = 0; k < quantity; k += sse_s) {
            // Set target values
            max_element   = _mm512_setzero_si512();
            max_element_i = _mm512_setzero_si512();
            max_element_j = _mm512_setzero_si512();

            // Construct next SIMDed batch of chars
            for (unsigned int i = 0; i < size; ++i) {
                for (unsigned int p = 0; p < sse_s; ++p) {
                    char_batch_i[p] = int(sequences[k + p].first[i]);
                    char_batch_j[p] = int(sequences[k + p].second[i]);
                }
                i_seq[i] = _mm512_load_si512( ( __m256i * ) char_batch_i );
                j_seq[i] = _mm512_load_si512( ( __m256i * ) char_batch_j );
            }
            
            for (unsigned int i = 1; i < size + 1; ++i){
                i_vectorized = _mm512_set1_epi32( i );
                for(unsigned int j = 1; j < size + 1; ++j) {
                    j_vectorized = _mm512_set1_epi32( j );
                    // match_val ~ data.first[i - 1] == data.second[j - 1] ? match : mismatch)
                    mask      = _mm512_cmpeq_epi32_mask( i_seq[i - 1], j_seq[j - 1] );
                    match_val = _mm512_mask_blend_epi32( mask, mismatch, match );
                    
                    // diagonal_value ~ matrix[i-1][j-1] + match_val
                    diagonal_value   = _mm512_add_epi32 ( matrix[i-1][j - 1], match_val );
                    // top_value ~ matrix[i-1][j] + gap
                    top_value        = _mm512_add_epi32 ( matrix[i-1][j], gap );
                    // left_value ~ matrix[i][j-1] + gap
                    left_value       = _mm512_add_epi32 ( matrix[i][j - 1], gap) ;
                    
                    // Calculate target_value ~ std::max(diagonal_value, std::max(top_value, left_value))
                    temp_value    = _mm512_max_epi32( top_value, left_value );
                    target_value  = _mm512_max_epi32( diagonal_value, temp_value );
                    // Calculate  matrix[i][j] ~ (target_value > 0) ? target_value : 0
                    matrix[i][j]  = _mm512_max_epi32( target_value, zeros );
                    // Update max_element and coordinates if the target_value is larger
                    max_element         = _mm512_max_epi32( max_element, target_value );
                    max_element_updated = _mm512_cmpeq_epi32_mask( max_element, target_value );
                    max_element_i       = _mm512_mask_blend_epi32( max_element_updated, max_element_i, i_vectorized );
                    max_element_j       = _mm512_mask_blend_epi32( max_element_updated, max_element_j, j_vectorized );
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
        #elif __AVX2__
        std::cout << "Using 256 bits wide registers ... " << std::endl;
        std::vector<std::vector<__m256i>> matrix(size + 1, std::vector<__m256i>(size + 1));

        // Instantiate SIMDed scores
        const __m256i gap      = _mm256_set1_epi32(-2);
        const __m256i mismatch =  _mm256_set1_epi32(-2);
        const __m256i match    =  _mm256_set1_epi32(3);

        // Target SIMDed values
        __m256i max_element, max_element_i, max_element_j;

        // Auxiliary values
        __m256i diagonal_value, top_value, left_value, temp_value,
                target_value, i_vectorized, j_vectorized, max_element_updated,
                mask, match_val;
        
        // SIMD size
        unsigned int sse_s     = 8;

        // Char batching containers
        std::vector<__m256i> i_seq( size );
        std::vector<__m256i> j_seq( size );
        int char_batch_i[ sse_s ];
        int char_batch_j[ sse_s ];

        for (unsigned int k = 0; k < quantity; k += sse_s) {
            // Set target values
            max_element   = _mm256_setzero_si256();
            max_element_i = _mm256_setzero_si256();
            max_element_j = _mm256_setzero_si256();

            // Construct next SIMDed batch of chars
            for (unsigned int i = 0; i < size; ++i) {
                for (unsigned int p = 0; p < sse_s; ++p) {
                    char_batch_i[p] = int(sequences[k + p].first[i]);
                    char_batch_j[p] = int(sequences[k + p].second[i]);
                }
                i_seq[i] = _mm256_load_si256( ( __m256i * ) char_batch_i );
                j_seq[i] = _mm256_load_si256( ( __m256i * ) char_batch_j );
            }
            
            for (unsigned int i = 1; i < size + 1; ++i){
                i_vectorized = _mm256_set1_epi32( i );
                for(unsigned int j = 1; j < size + 1; ++j) {
                    j_vectorized = _mm256_set1_epi32( j );
                    // match_val ~ data.first[i - 1] == data.second[j - 1] ? match : mismatch)
                    mask      = _mm256_cmpeq_epi32( i_seq[i - 1], j_seq[j - 1] );
                    match_val = _mm256_blendv_epi8( mismatch, match, mask );
                    
                    // diagonal_value ~ matrix[i-1][j-1] + match_val
                    diagonal_value   = _mm256_add_epi32 ( matrix[i-1][j - 1], match_val );
                    // top_value ~ matrix[i-1][j] + gap
                    top_value        = _mm256_add_epi32 ( matrix[i-1][j], gap );
                    // left_value ~ matrix[i][j-1] + gap
                    left_value       = _mm256_add_epi32 ( matrix[i][j - 1], gap) ;
                    
                    // Calculate target_value ~ std::max(diagonal_value, std::max(top_value, left_value))
                    temp_value    = _mm256_max_epi32( top_value, left_value );
                    target_value  = _mm256_max_epi32( diagonal_value, temp_value );
                    // Calculate  matrix[i][j] ~ (target_value > 0) ? target_value : 0
                    matrix[i][j]  = _mm256_max_epi32( target_value, _mm256_setzero_si256() );
                    // Update max_element and coordinates if the target_value is larger
                    max_element         = _mm256_max_epi32( max_element, target_value );
                    max_element_updated = _mm256_cmpeq_epi32( max_element, target_value );
                    max_element_i       = _mm256_blendv_epi8( max_element_i, i_vectorized, max_element_updated );
                    max_element_j       = _mm256_blendv_epi8( max_element_j, j_vectorized, max_element_updated );
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
        #else
        std::cout << "Your CPU supprots SIMD intructions that are required to run this code. This implementation expects either AVX2 or AVX512 support." << std::endl;
        #endif
    }

