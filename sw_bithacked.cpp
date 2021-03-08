#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include "traceback.cpp"


template < typename T >
    void sw_bithacked(std::pair< T, T > data){
        // instantiate a matrix 
        std::vector<std::vector<int>> matrix(data.first.size() + 1,std::vector<int>(data.second.size() + 1,0));
        
        // populate the matrix
        int gaps(-2), mismatch(-2), match(3);
        int max_element(0), max_element_i(0), max_element_j(0);
        int diagonal_value, top_value, left_value;
        for (long unsigned int i=1; i<data.first.size() + 1; ++i){
            for(long unsigned int j=1; j<data.second.size() + 1; ++j) {
                    diagonal_value = matrix[i-1][j - 1] + (data.first[i - 1] == data.second[j - 1] ? match : mismatch);
                    top_value = matrix[i-1][j] + gaps;
                    left_value = matrix[i][j - 1] + gaps;
                    int temp = top_value - ((top_value - left_value) & ((top_value - left_value) >> (sizeof(int) * 8 - 1)));
                    int target_value = diagonal_value - ((diagonal_value - temp) & ((diagonal_value - temp) >> (sizeof(int) * 8 - 1)));  // std::max(diagonal_value, std::max(top_value, left_value));
                    matrix[i][j] = target_value - (target_value & (target_value >> (sizeof(int) * 8 - 1)));  // (target_value > 0) ? target_value : 0;
                    // Bithacks to replace branching below:
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
        std::cout << max_element << " " << max_element_i << " " << max_element_j << std::endl;

        // // Traceback
        // traceback(matrix, max_element_i, max_element_j);
}