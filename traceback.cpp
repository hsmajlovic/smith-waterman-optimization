#ifndef TRACEBACK
#define TRACEBACK

#include <vector>
#include <string>
#include <iostream>


template < typename T >
void traceback(
        std::pair<std::string, std::string> sequence_pair,
        std::vector<std::vector<T>> matrix,
        int max_element_i, int max_element_j,
        int match, int mismatch, int gap) {
    std::vector<std::pair<int, int>> traceback_indices;
    std::string alignment_str_1("");
    std::string alignment_str_2("");
    int current_i = max_element_i;
    int current_j = max_element_j;

    T diagonal_value, top_value, left_value;

    while (matrix[current_i][current_j]) {
        traceback_indices.push_back(std::make_pair(current_i, current_j));
        diagonal_value = matrix[current_i][current_j] - (sequence_pair.first[current_i - 1] == sequence_pair.second[current_j - 1] ? match : mismatch);
        top_value = matrix[current_i][current_j] - gap;
        left_value = matrix[current_i][current_j] - gap;
        if (diagonal_value == matrix[current_i-1][current_j-1]) {
            current_i = current_i - 1;
            current_j = current_j - 1;
            alignment_str_1 += sequence_pair.first[current_i];
            alignment_str_2 += sequence_pair.second[current_j];
        } else if (top_value == matrix[current_i-1][current_j]) {
            current_i = current_i - 1;
            alignment_str_1 += sequence_pair.first[current_i];
            alignment_str_2 += '-';
        } else {
            current_j = current_j - 1;
            alignment_str_1 += '-';
            alignment_str_2 += sequence_pair.second[current_j];
        }
    }

    std::reverse(alignment_str_1.begin(), alignment_str_1.end());
    std::reverse(alignment_str_2.begin(), alignment_str_2.end());

    std::cout << alignment_str_1 << " " << alignment_str_2 << std::endl;
}
#endif
