#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include "utils.cpp"

#include <stdio.h>


using dp_mat = int[ SIZE ][ SIZE ];

__global__
void align(int *scores, dp_mat *matrices, char *sequences, size_t size) {
    // Thread index
    const int t = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Instantiate scores
    auto const gap      = -2;
    auto const mismatch = -2;
    auto const match    = 3;

    // Instantiate placeholders
    int max_element(0),
        // max_element_i(0),
        // max_element_j(0),
        diagonal_value,
        top_value,
        left_value;
    
    for ( auto i = 1; i < size + 1; ++i ) {
        for( auto j = 1; j < size + 1; ++j ) {
            diagonal_value = matrices[ t ][ i-1 ][ j - 1 ];
            diagonal_value += (
                sequences[ t * size * 2 + i - 1 ] == sequences[t * size * 2 + j - 1 + size] ? match : mismatch);
            top_value = matrices[ t ][ i-1 ][ j ] + gap;
            left_value = matrices[ t ][ i ][ j - 1 ] + gap;
            int temp = top_value - ((top_value - left_value) & ((top_value - left_value) >> (sizeof(int) * 8 - 1)));
            int target_value = diagonal_value - ((diagonal_value - temp) & ((diagonal_value - temp) >> (sizeof(int) * 8 - 1)));  // std::max(diagonal_value, std::max(top_value, left_value));
            matrices[ t ][ i ][ j ] = target_value - (target_value & (target_value >> (sizeof(int) * 8 - 1)));  // (target_value > 0) ? target_value : 0;
            // Bithacks to replace branching below:
            // max_element = target_value - ((target_value - max_element) & ((target_value - max_element) >> (sizeof(int) * 8 - 1)));
            // temp = int(target_value != max_element);
            // int temp_i = temp * int(i);
            // max_element_i = max_element_i - ((max_element_i - temp_i) & ((max_element_i - temp_i) >> (sizeof(int) * 8 - 1)));
            // int temp_j = temp * int(j);
            // max_element_j = max_element_j - ((max_element_j - temp_j) & ((max_element_j - temp_j) >> (sizeof(int) * 8 - 1)));
            if ( target_value > max_element ) {
                max_element = target_value;
                // max_element_i = i;
                // max_element_j = j;
            }
        }
    }

    scores[t] = max_element;

    // // Traceback
    // traceback(matrix, max_element_i, max_element_j);
}

void sw_cuda_alpern(std::vector<std::pair<std::string, std::string>> const sequences){
    // Quanitities
    int const num_blocks = QUANTITY / CUDA_BLOCK_SIZE;
    
    // Instantiate host variables
    std::vector<int> scores(QUANTITY);
    
    // Instantiate device variables
    dp_mat *dev_matrices;
    char   *dev_input;
    int    *dev_output;
    int matrices_size = QUANTITY * sizeof(dp_mat);
    int output_size   = QUANTITY * sizeof(int);
    int input_size    = QUANTITY * SIZE * 2;

    // Allocate memory on device
    cudaMalloc( (void**) &dev_output, output_size );
    cudaMalloc( (void**) &dev_matrices, matrices_size );
    cudaMalloc( (void**) &dev_input, input_size );

    // Send the data to device
    cudaMemcpy( dev_input, to_byte_arr(sequences), input_size, cudaMemcpyHostToDevice );
    
    // Kernel
    align<<< num_blocks, CUDA_BLOCK_SIZE>>>(dev_output, dev_matrices, dev_input, SIZE);

    // Retrieve results from device
    cudaMemcpy( scores.data(), dev_output, output_size, cudaMemcpyDeviceToHost );

    // Free the memory on device
    cudaFree( dev_input );
    cudaFree( dev_matrices );
    cudaFree( dev_output );
}
