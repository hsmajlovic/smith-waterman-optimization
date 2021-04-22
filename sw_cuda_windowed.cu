#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include "utils.h"

#include <stdio.h>


using window_t = int16_t[ WINDOW_SIZE ][ WINDOW_SIZE ];

__global__
void align(int16_t *scores, window_t *windows, char *sequences) {
    // Thread index
    const int16_t t = threadIdx.x + blockIdx.x * blockDim.x;

    // Instantiate scores
    auto const gap      = -2;
    auto const mismatch = -2;
    auto const match    = 3;

    // Calculate params
    const int16_t batches_no = SIZE / WINDOW_SIZE;
    int16_t top_leftover[WINDOW_SIZE] = { };
    int16_t side_leftover[WINDOW_SIZE] = { };

    // Instantiate placeholders
    int16_t max_element(0),
        diagonal_value,
        top_value,
        left_value;
    char s1_piece[WINDOW_SIZE];
    char s2_piece[WINDOW_SIZE];
    
    for (int16_t bi = 0; bi < batches_no; ++bi) {
        int16_t offset_i = bi * WINDOW_SIZE;
        
        for (int16_t i = 0; i != WINDOW_SIZE; ++i)
            s1_piece[ i ] = sequences[ t * SIZE * 2 + offset_i + i ];
        
        for (int16_t bj = 0; bj < batches_no; ++bj) {
            int16_t offset_j = bj * WINDOW_SIZE;
            
            for (int16_t i = 0; i != WINDOW_SIZE; ++i)
                s2_piece[ i ] = sequences[ t * SIZE * 2 + SIZE + offset_j + i ];
            
            for (int16_t i = 0; i < WINDOW_SIZE; ++i) {
                windows[ t ][ 0 ][ i ] = top_leftover[ i ];
                windows[ t ][ i ][ 0 ] = side_leftover[ i ];
            }
            
            for (int16_t i = 1; i < WINDOW_SIZE; ++i) {
                for (int16_t j = 1; j < WINDOW_SIZE; ++j) {
                    diagonal_value = windows[ t ][ i - 1 ][ j - 1 ] + (s1_piece[i - 1] == s2_piece[j - 1] ? match : mismatch);
                    top_value = windows[ t ][ i - 1 ][ j ] + gap;
                    left_value = windows[ t ][ i ][ j - 1 ] + gap;
                    // target_value = std::max(diagonal_value, std::max(top_value, left_value));
                    int16_t temp =
                        top_value - ((top_value - left_value) & ((top_value - left_value) >> (sizeof(int16_t) * 8 - 1)));
                    int16_t target_value =
                        diagonal_value - ((diagonal_value - temp) & ((diagonal_value - temp) >> (sizeof(int16_t) * 8 - 1)));
                    // windows[ t ][ i ][ j ] = (target_value > 0) ? target_value : 0;
                    windows[ t ][ i ][ j ] =
                        target_value - (target_value & (target_value >> (sizeof(int16_t) * 8 - 1)));
                    if (target_value > max_element)
                        max_element = target_value;
                }
            }

            for (int16_t i = 0; i < WINDOW_SIZE; ++i) {
                top_leftover[ i ] = windows[ t ][ WINDOW_SIZE - 1 ][ i ];
                side_leftover[ i ] = windows[ t ][ i ][ WINDOW_SIZE - 1 ];
            }
        }
    }

    scores[ t ] = max_element;

    // // Traceback
    // traceback(matrix, max_element_i, max_element_j);
}

void sw_cuda_windowed(std::vector<std::pair<std::string, std::string>> const sequences){
    // Quanitities
    int const num_blocks = QUANTITY / CUDA_BLOCK_SIZE;
    
    // Instantiate host variables
    std::vector<int16_t> scores(QUANTITY);
    
    // Instantiate device variables
    window_t  *dev_matrices;
    char      *dev_input;
    int16_t   *dev_output;
    int64_t matrices_size = QUANTITY * sizeof(window_t);
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
    align<<< num_blocks, CUDA_BLOCK_SIZE>>>( dev_output, dev_matrices, dev_input );
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
