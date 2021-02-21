#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include "smith_waterman.h"


template < typename T >
    void smith_waterman_striped(std::pair< T, T > sequences){
        // TODO: Switch everything to unsigned ints
        
        // instantiate a matrix 
        T s1 = sequences.first;
        T s2 = sequences.second;
        unsigned int size = s1.size();
        std::vector<std::vector<int>> matrix(size + 1, std::vector<int>(size + 1, 0));

        int gaps(-2), mismatch(-2), match(3), max_element(0);
        unsigned int max_element_i(0), max_element_j(0);
        int diagonal_value, top_value, left_value;
        
        unsigned int stripe = 1u << 8;
        for (unsigned int s=1; s < size + 1; s += stripe) {
            for (unsigned int i=1; i < size + 1; ++i){
                for(unsigned int j=0; j < stripe; ++j) {
                    diagonal_value = matrix[i-1][s + j - 1] + (s1[i - 1] == s2[s + j - 1] ? match : mismatch);
                    top_value = matrix[i-1][s + j] + gaps;
                    left_value = matrix[i][s + j - 1] + gaps;
                    int temp = top_value - ((top_value - left_value) & ((top_value - left_value) >> (sizeof(int) * 8 - 1)));
                    int target_value = diagonal_value - ((diagonal_value - temp) & ((diagonal_value - temp) >> (sizeof(int) * 8 - 1)));  // std::max(diagonal_value, std::max(top_value, left_value));
                    matrix[i][s + j] = target_value - (target_value & (target_value >> (sizeof(int) * 8 - 1)));  // (target_value > 0) ? target_value : 0;
                    // max_element = target_value - ((target_value - max_element) & ((target_value - max_element) >> (sizeof(int) * 8 - 1)));
                    // temp = int(target_value != max_element);
                    // int temp_i = temp * int(i);
                    // max_element_i = max_element_i - ((max_element_i - temp_i) & ((max_element_i - temp_i) >> (sizeof(int) * 8 - 1)));
                    // int temp_j = temp * int(j);
                    // max_element_j = max_element_j - ((max_element_j - temp_j) & ((max_element_j - temp_j) >> (sizeof(int) * 8 - 1)));
                    if (target_value > max_element) {
                        max_element = target_value;
                        max_element_i = i;
                        max_element_j = j;
                    }
                }
            }
        }
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
