#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include "utils.h"

#include <stdio.h>


using dp_mat = int16_t[ SIZE + 1 ][ SIZE + 1 ];

__global__
void align(int16_t *scores, dp_mat *matrices, char *sequences, size_t size) {
    // Thread index
    const int16_t t = threadIdx.x + blockIdx.x * blockDim.x;

    // Instantiate scores
    auto const gap      = -2;
    auto const mismatch = -2;
    auto const match    = 3;

    // Instantiate placeholders
    int16_t max_element(0),
        // max_element_i(0),
        // max_element_j(0),
        diagonal_value,
        top_value,
        left_value;
    
    for ( int16_t i = 1; i < size + 1; ++i ) {
        for( int16_t j = 1; j < size + 1; ++j ) {
            diagonal_value = matrices[ t ][ i - 1 ][ j - 1 ];
            diagonal_value += (
                sequences[ t * size * 2 + i - 1 ] == sequences[ t * size * 2 + j - 1 + size ] ? match : mismatch);
            top_value = matrices[ t ][ i - 1 ][ j ] + gap;
            left_value = matrices[ t ][ i ][ j - 1 ] + gap;
            
            // diagonal_value = std::max(diagonal_value, std::max(top_value, left_value));
            int16_t temp =
                top_value - ((top_value - left_value) & ((top_value - left_value) >> (sizeof(int16_t) * 8 - 1)));
            int16_t target_value =
                diagonal_value - ((diagonal_value - temp) & ((diagonal_value - temp) >> (sizeof(int16_t) * 8 - 1)));
            // matrices[ t ][ i ][ j ] = (target_value > 0) ? target_value : 0;
            matrices[ t ][ i ][ j ] =
                target_value - (target_value & (target_value >> (sizeof(int16_t) * 8 - 1)));
            // // Bithacks to replace branching below:
            // max_element = target_value - ((target_value - max_element) & ((target_value - max_element) >> (sizeof(int16_t) * 8 - 1)));
            // temp = int16_t(target_value != max_element);
            // int16_t temp_i = temp * int16_t(i);
            // max_element_i = max_element_i - ((max_element_i - temp_i) & ((max_element_i - temp_i) >> (sizeof(int16_t) * 8 - 1)));
            // int16_t temp_j = temp * int16_t(j);
            // max_element_j = max_element_j - ((max_element_j - temp_j) & ((max_element_j - temp_j) >> (sizeof(int16_t) * 8 - 1)));
                
            if ( target_value > max_element ) {
                max_element = target_value;
                // max_element_i = i;
                // max_element_j = j;
            }
        }
    }

    scores[ t ] = max_element;

    // // Traceback
    // traceback(matrix, max_element_i, max_element_j);
}

void sw_cuda_alpern(std::vector<std::pair<std::string, std::string>> const sequences){
    // Quanitities
    int const num_blocks = QUANTITY / CUDA_BLOCK_SIZE;
    
    // Instantiate host variables
    std::vector<int16_t> scores(QUANTITY);
    
    // Instantiate device variables
    dp_mat  *dev_matrices;
    char    *dev_input;
    int16_t *dev_output;
    int64_t matrices_size = QUANTITY * sizeof(dp_mat);
    int64_t output_size   = QUANTITY * sizeof(int16_t);
    int64_t input_size    = QUANTITY * SIZE * 2;

    // Allocate memory on device
    auto const start_time_1 = std::chrono::steady_clock::now();
    cudaMalloc( (void**) &dev_output, output_size );
    cudaMalloc( (void**) &dev_matrices, matrices_size );
    cudaMalloc( (void**) &dev_input, input_size );
    auto const end_time_1 = std::chrono::steady_clock::now();
	std::cout << "Device malloc time: (μs) "
              << std::chrono::duration_cast<std::chrono::microseconds>( end_time_1 - start_time_1 ).count()
              << std::endl;
    cudaCheckErrors("Failed to allocate device buffer");

    // Preprocessing
    auto const start_time_0 = std::chrono::steady_clock::now();
    const char *sequences_bytes = to_byte_arr(sequences);
    auto const end_time_0 = std::chrono::steady_clock::now();
	std::cout << "Preprocessing time: (μs) "
              << std::chrono::duration_cast<std::chrono::microseconds>( end_time_0 - start_time_0 ).count()
              << std::endl;
    
    // Send the data to device
    auto const start_time_2 = std::chrono::steady_clock::now();
    cudaMemcpy( dev_input, sequences_bytes, input_size, cudaMemcpyHostToDevice );
    cudaDeviceSynchronize();
    auto const end_time_2 = std::chrono::steady_clock::now();
	std::cout << "To device transfer time: (μs) "
              << std::chrono::duration_cast<std::chrono::microseconds>( end_time_2 - start_time_2 ).count()
              << std::endl;
    cudaCheckErrors("CUDA memcpy failure");
    
    // Kernel
    auto const start_time_3 = std::chrono::steady_clock::now();
    align<<< num_blocks, CUDA_BLOCK_SIZE>>>( dev_output, dev_matrices, dev_input, SIZE );
    cudaDeviceSynchronize();
    auto const end_time_3 = std::chrono::steady_clock::now();
    std::cout << "Exec time: (μs) "
              << std::chrono::duration_cast<std::chrono::microseconds>( end_time_3 - start_time_3 ).count()
              << std::endl;
    cudaCheckErrors("Kernel launch failure");

    // Retrieve results from device
    auto const start_time_4 = std::chrono::steady_clock::now();
    cudaMemcpy( scores.data(), dev_output, output_size, cudaMemcpyDeviceToHost );
    auto const end_time_4 = std::chrono::steady_clock::now();
    std::cout << "From device transfer time: (μs) "
              << std::chrono::duration_cast<std::chrono::microseconds>( end_time_4 - start_time_4 ).count()
              << std::endl;
    cudaCheckErrors("CUDA memcpy failure");

    // Free the memory on device
    cudaFree( dev_input );
    cudaFree( dev_matrices );
    cudaFree( dev_output );
    cudaCheckErrors("cudaFree fail");

    int total_scores;
    for (auto e : scores) {
        total_scores += e;
    }
    std::cout << "Average score: " << total_scores / QUANTITY << std::endl;
}
