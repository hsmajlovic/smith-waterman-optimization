#include <iostream>  // std::cout
#include <string>
#include <vector>
#include <utility>
#include "utils.h"

#include <stdio.h>


using dp_mat = int16_t[ SIZE + 1 ][ SIZE + 1 ];

__device__
void atomic_kernel(int *scores, dp_mat *matrices, char *sequences,
                   int16_t tid, int16_t i, int16_t j,
                   int16_t gap, int16_t mismatch, int16_t match) {
    int16_t diagonal_value = matrices[ tid ][ i - 1 ][ j - 1 ];
    diagonal_value += (
        sequences[ tid * SIZE * 2 + i - 1 ] == sequences[ tid * SIZE * 2 + j - 1 + SIZE ] ? match : mismatch);
    int16_t top_value = matrices[ tid ][ i - 1 ][ j ] + gap;
    int16_t left_value = matrices[ tid ][ i ][ j - 1 ] + gap;
    
    // diagonal_value = std::max(diagonal_value, std::max(top_value, left_value));
    int16_t temp =
        top_value - ((top_value - left_value) & ((top_value - left_value) >> (sizeof(int16_t) * 8 - 1)));
    int16_t target_value =
        diagonal_value - ((diagonal_value - temp) & ((diagonal_value - temp) >> (sizeof(int16_t) * 8 - 1)));
    // matrices[ t ][ i ][ j ] = (target_value > 0) ? target_value : 0;
    matrices[ tid ][ i ][ j ] =
        target_value - (target_value & (target_value >> (sizeof(int16_t) * 8 - 1)));
    
    atomicMax(scores + tid, int(target_value));
}

__global__
void antidiagonal_kernel(int *scores, dp_mat *matrices, char *sequences) {
    // Indices
    const int16_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int16_t rid = threadIdx.y + blockIdx.y * blockDim.y;

    if (tid > QUANTITY || rid > SIZE)
        return;

    // Instantiate scores
    auto const gap      = -2;
    auto const mismatch = -2;
    auto const match    = 3;
    
    for (int16_t did = 0; did != rid + SIZE; ++did) {
        int16_t row = rid + 1;
        int16_t column = (did < rid ? did : did - rid) + 1;
        atomic_kernel(scores, matrices, sequences,
                      tid, row, column,
                      gap, mismatch, match);
        __syncthreads();
    }
}


void sw_cuda_antidiagonal(std::vector<std::pair<std::string, std::string>> const sequences){
    // Instantiate host variables
    std::vector<int> scores(QUANTITY);
    
    // Instantiate device variables
    dp_mat  *dev_matrices;
    char    *dev_input;
    int     *dev_output;
    int64_t matrices_size = QUANTITY * sizeof(dp_mat);
    int64_t output_size   = QUANTITY * sizeof(int);
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
    
    auto const end_time_2 = std::chrono::steady_clock::now();
	std::cout << "To device transfer time: (μs) "
              << std::chrono::duration_cast<std::chrono::microseconds>( end_time_2 - start_time_2 ).count()
              << std::endl;
    cudaCheckErrors("CUDA memcpy failure");
    
    // Kernel
    auto const start_time_3 = std::chrono::steady_clock::now();
    
    dim3 blockSize(CUDA_XBLOCK_SIZE, CUDA_YBLOCK_SIZE);
    dim3 gridSize(QUANTITY / CUDA_XBLOCK_SIZE, SIZE / CUDA_YBLOCK_SIZE);
    antidiagonal_kernel<<< gridSize, blockSize >>>( dev_output, dev_matrices, dev_input );
    
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
