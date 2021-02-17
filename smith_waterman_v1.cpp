#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include "smith_waterman.h"


template < typename T >
    void smith_waterman_v1(std::pair< T, T > sequences){
        // TODO: Switch everything to unsigned ints
        
        // instantiate a matrix 
        T s1 = sequences.first;
        T s2 = sequences.second;
        unsigned int size = s1.size();
        // std::vector<std::vector<int>> matrix(size + 1, std::vector<int>(size + 1, 0));

        unsigned int window_size = 1u << 6;
        unsigned int batches_no = size / window_size;
        std::vector<int> top_leftover(window_size, 0);
        std::vector<int> side_leftover(window_size, 0);
        std::vector<std::vector<int>> window(window_size + 1, std::vector<int>(window_size + 1, 0));

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
                    window[0][i] = top_leftover[i];
                    window[i][0] = side_leftover[i];
                }
                
                for (unsigned int i = 1; i < window_size; ++i) {
                    for (unsigned int j = 1; j < window_size; ++j) {
                        diagonal_value = window[i-1][j-1] + (s1_piece[i - 1] == s2_piece[j - 1] ? match : mismatch);
                        top_value = window[i-1][j] + gaps;
                        left_value = window[i][j-1] + gaps;
                        int temp = top_value - ((top_value - left_value) & ((top_value - left_value) >> (sizeof(int) * 8 - 1)));
                        int target_value = diagonal_value - ((diagonal_value - temp) & ((diagonal_value - temp) >> (sizeof(int) * 8 - 1)));  // std::max(diagonal_value, std::max(top_value, left_value));
                        window[i][j] = target_value - (target_value & (target_value >> (sizeof(int) * 8 - 1)));  // (target_value > 0) ? target_value : 0;
                        // max_element = target_value - ((target_value - max_element) & ((target_value - max_element) >> (sizeof(int) * 8 - 1)));
                        // temp = int(target_value != max_element);
                        // int temp_i = temp * int(i);
                        // max_element_i = max_element_i - ((max_element_i - temp_i) & ((max_element_i - temp_i) >> (sizeof(int) * 8 - 1)));
                        // int temp_j = temp * int(j);
                        // max_element_j = max_element_j - ((max_element_j - temp_j) & ((max_element_j - temp_j) >> (sizeof(int) * 8 - 1)));
                        if (target_value > max_element) {
                            max_element = target_value;
                            max_element_i = offset_i + i;
                            max_element_j = offset_j + j;
                        }
                    }
                }

                // for (unsigned int i = 1; i < window_size; ++i) {
                //     for (unsigned int j = 1; j < window_size; ++j) {
                //         matrix[offset_i + i][offset_j + j] = window[i][j];
                //     }
                // }

                for (unsigned int i = 0; i < window_size; ++i) {
                    top_leftover[i] = window[window_size - 1][i];
                    side_leftover[i] = window[i][window_size - 1];
                }
            }
        }


        // // populate the matrix
        // for (unsigned int i=1; i<size1 + 1; ++i){
        //     for(unsigned int j=1; j<size2 + 1; ++j) {
        //         // Alternative: Declare here
        //         diagonal_value = matrix[i-1][j-1] + (s1[i - 1] == s2[j - 1] ? match : mismatch);
        //         top_value = matrix[i-1][j] + gaps;
        //         left_value = matrix[i][j-1] + gaps;
        //         int temp = top_value - ((top_value - left_value) & ((top_value - left_value) >> (sizeof(int) * 8 - 1)));
        //         int target_value = diagonal_value - ((diagonal_value - temp) & ((diagonal_value - temp) >> (sizeof(int) * 8 - 1)));  // std::max(diagonal_value, std::max(top_value, left_value));
        //         matrix[i][j] = target_value - (target_value & (target_value >> (sizeof(int) * 8 - 1)));  // (target_value > 0) ? target_value : 0;
        //         // max_element = target_value - ((target_value - max_element) & ((target_value - max_element) >> (sizeof(int) * 8 - 1)));
        //         // temp = int(target_value != max_element);
        //         // int temp_i = temp * int(i);
        //         // max_element_i = max_element_i - ((max_element_i - temp_i) & ((max_element_i - temp_i) >> (sizeof(int) * 8 - 1)));
        //         // int temp_j = temp * int(j);
        //         // max_element_j = max_element_j - ((max_element_j - temp_j) & ((max_element_j - temp_j) >> (sizeof(int) * 8 - 1)));
        //         if (target_value > max_element) {
        //             max_element = target_value;
        //             max_element_i = i;
        //             max_element_j = j;
        //         }
        //     }
        // }

        std::cout << max_element << " " << max_element_i << " " << max_element_j << std::endl;


        // traceback

        // std::vector<std::pair<int, int>> traceback_indices;
        // std::string alignment_str_1("");
        // std::string alignment_str_2("");
        // int current_i = max_element_i;
        // int current_j = max_element_j;
        // while (matrix[current_i][current_j]) {
        //     traceback_indices.push_back(std::make_pair(current_i, current_j));
        //     diagonal_value = matrix[current_i][current_j] - (data.first[current_i - 1] == data.second[current_j - 1] ? match : mismatch);
        //     top_value = matrix[current_i][current_j] - gaps;
        //     left_value = matrix[current_i][current_j] - gaps;
        //     if (diagonal_value == matrix[current_i-1][current_j-1]) {
        //         current_i = current_i - 1;
        //         current_j = current_j - 1;
        //         alignment_str_1 += data.first[current_i];
        //         alignment_str_2 += data.second[current_j];
        //     } else if (top_value == matrix[current_i-1][current_j]) {
        //         current_i = current_i - 1;
        //         alignment_str_1 += data.first[current_i];
        //         alignment_str_2 += '-';
        //     } else {
        //         current_j = current_j - 1;
        //         alignment_str_1 += '-';
        //         alignment_str_2 += data.second[current_j];
        //     }
        // }

        // std::reverse(alignment_str_1.begin(), alignment_str_1.end());
        // std::reverse(alignment_str_2.begin(), alignment_str_2.end());

        // std::cout << alignment_str_1 << " " << alignment_str_2 << std::endl;

        // std::copy(
        //     traceback_indices.begin(), traceback_indices.end(),
        //     std::ostream_iterator<std::pair<int, int>>(std::cout, " "));
}