#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator


template < typename T >
    void smith_waterman_base(std::pair< T, T > data){
        // instantiate a matrix 
        std::vector<std::vector<int>> matrix(data.first.size() + 1,std::vector<int>(data.second.size() + 1,0));
        
        // populate the matrix
        int gaps(-2), mismatch(-2), match(3);
        int diagonal_value, top_value, left_value;
        for (long unsigned int i=1; i<data.first.size() + 1; ++i){
            for(long unsigned int j=1; j<data.second.size() + 1; ++j) {
                // Alternative: Declare here
                diagonal_value = matrix[i-1][j-1] + (data.first[i - 1] == data.second[j - 1] ? match : mismatch);
                top_value = matrix[i-1][j] + gaps;
                left_value = matrix[i][j-1] + gaps;
                int target_value = std::max(diagonal_value, std::max(top_value, left_value));
                matrix[i][j] = (target_value > 0) ? target_value : 0;
            }
        }

        // // traceback
        // // Largest value indices
        // int max_element(0), max_element_i(0), max_element_j(0);
        // for (long unsigned int i=1; i<data.first.size() + 1; ++i) {
        //     std::vector<int>::iterator row_max_iterator = std::max_element(matrix[i].begin(), matrix[i].end());
        //     if (*row_max_iterator > max_element) {
        //         max_element_i = i;
        //         max_element_j = row_max_iterator - matrix[i].begin();
        //         max_element = *row_max_iterator;
        //     }
        // }

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
}