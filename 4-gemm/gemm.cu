
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Custom batched GEMM
//
// C[b] = A[b] * B[b]
//
// Matrices are N x N, column-major.
////////////////////////////////////////////////////////////////////////////////
__global__ void batch_gemm_kernel(int n, int batch_size,
                                    const float* A,
                                    const float* B,
                                    float* C)
{
    // TODO: write your own kernel here
    // Identification de la position du thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int b   = blockIdx.z;

    // Vérification des limites
    if (row < n && col < n && b < batch_size)
    {
        size_t matrix_size = (size_t)n * n;
        size_t base = (size_t)b * matrix_size;

        float sum = 0.0f;

        // Calcul du produit scalaire
        for (int k = 0; k < n; ++k)
        {
            sum += A[base + (size_t)k * n + row]
                 * B[base + (size_t)col * n + k];
        }

        // Stockage du résultat
        C[base + (size_t)col * n + row] = sum;
    }
}



////////////////////////////////////////////////////////////////////////////////
// CPU reference implementation
void batched_gemm_cpu(int n, int batch_size,
                      const float* A,
                      const float* B,
                      float* C)
{
    const size_t matrix_size = (size_t)n * n;

    for (int b = 0; b < batch_size; ++b) {

        const size_t base = (size_t)b * matrix_size;

        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < n; ++row) {

                float sum = 0.0f;

                for (int k = 0; k < n; ++k) {
                    sum += A[base + (size_t)k * n + row] *
                           B[base + (size_t)col * n + k];
                }

                C[base + (size_t)col * n + row] = sum;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Forward error for one matrix
//
// ||Cgpu-Cref||_F / (N ||Cref||_F)
// ------------------------------------------------------------
double forward_error(int n,
                     const float* result,
                     const float* reference)
{
    double diff2 = 0.0;
    double ref2  = 0.0;

    const size_t size = (size_t)n * n;

    for (size_t i = 0; i < size; ++i) {
        const double diff =
            (double)result[i] - (double)reference[i];

        diff2 += diff * diff;
        ref2  += (double)reference[i] * reference[i];
    }

    const double diff_norm = std::sqrt(diff2);
    const double ref_norm  = std::sqrt(ref2);

    if (ref_norm > 0.0)
        return diff_norm / (n * ref_norm);

    return (diff_norm == 0.0) ? 0.0 : INFINITY;
}

////////////////////////////////////////////////////////////////////////////////
// Maximum forward error over the batch
double max_batch_forward_error(int n,
                               int batch_size,
                               const float* result,
                               const float* reference)
{
    const size_t matrix_size = (size_t)n * n;

    double max_error = 0.0;

    for (int b = 0; b < batch_size; ++b) {

        const size_t base = (size_t)b * matrix_size;

        const double e =
            forward_error(n,
                          result + base,
                          reference + base);

        max_error = std::max(max_error, e);
    }

    return max_error;
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    const int n          = (argc > 1) ? std::atoi(argv[1]) : 16;
    const int batch_size = (argc > 2) ? std::atoi(argv[2]) : 1000;
    const int repeats = 100;

    if (n <= 0 || n > 32 || batch_size <= 0) {

        std::fprintf(stderr, "Usage: %s matrix_size(1..32) batch_size\n", argv[0]);

        return EXIT_FAILURE;
    }

    const size_t matrix_size = (size_t)n * n;
    const size_t total = matrix_size * batch_size;
    const size_t bytes = total * sizeof(float);

    // --------------------------------------------------------
    // Initialize input matrices
    // --------------------------------------------------------
    std::vector<float> h_A(total);
    std::vector<float> h_B(total);
    std::vector<float> h_ref(total);
    std::vector<float> h_custom(total);
    std::vector<float> h_cublas(total);

    for (size_t i = 0; i < total; ++i) {
        h_A[i] = std::sin(0.013f * (float)(i + 1));
        h_B[i] = std::cos(0.017f * (float)(i + 1));
    }

    // ref. CPU GEMM
    batched_gemm_cpu(n, batch_size, h_A.data(), h_B.data(), h_ref.data());

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------
    // TODO: write your allocations here

    // Pointeurs vers la mémoire GPU
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    // Allocation de la mémoire GPU
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);


    // --------------------------------------------------------
    // Copy A and B to GPU
    // --------------------------------------------------------
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // --------------------------------------------------------
    // GPU information
    // --------------------------------------------------------
    int device = 0;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::printf("GPU: %s (compute capability %d.%d)\n",
                prop.name, prop.major, prop.minor);

    std::printf("Matrix size: %d x %d, batch size: %d\n",
                n, n, batch_size);

    std::printf("FP32, column-major matrices, repeats: %d\n\n",
                repeats);

    // --------------------------------------------------------
    // CUDA events
    // --------------------------------------------------------
    // Création des événements CUDA
      cudaEvent_t start, stop;

      cudaEventCreate(&start);
      cudaEventCreate(&stop);

    // --------------------------------------------------------
    // Custom kernel configuration
    // --------------------------------------------------------
    // Configuration du kernel
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16,
          (n + 15) / 16,
          batch_size);


    // --------------------------------------------------------
    // Warmup run
    // --------------------------------------------------------
     // Exécution initiale sans mesure du temps
    batch_gemm_kernel<<<grid, block>>>(
            n, batch_size, d_A, d_B, d_C
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

err = cudaDeviceSynchronize();
if (err != cudaSuccess) {
    printf("Kernel execution error: %s\n", cudaGetErrorString(err));
}
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    for (int r = 0; r < repeats; ++r) {
        batch_gemm_kernel<<<grid, block>>>(
              n, batch_size, d_A, d_B, d_C
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float custom_ms = 0.0f;
    cudaEventElapsedTime(&custom_ms, start, stop);

    const double custom_us = custom_ms * 1000.0 / repeats;


    cudaMemcpy(
    h_custom.data(),
    d_C,
    bytes,
    cudaMemcpyDeviceToHost
  );
  printf("Custom result[0] = %f\n", h_custom[0]);
  printf("CPU reference[0] = %f\n", h_ref[0]);

    // --------------------------------------------------------
    // cuBLAS strided batched GEMM
    // --------------------------------------------------------

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_A, n, (long long)matrix_size,
        d_B, n, (long long)matrix_size,
        &beta,
        d_C, n, (long long)matrix_size,
        batch_size
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS error: %d\n", (int)status);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int r = 0; r < repeats; ++r) {

       cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n, n, n,
            &alpha,
            d_A, n, (long long)matrix_size,
            d_B, n, (long long)matrix_size,
            &beta,
            d_C, n, (long long)matrix_size,
            batch_size
     );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cublas_ms = 0.0f;
   cudaEventElapsedTime(&cublas_ms, start, stop);
    if (time_err != cudaSuccess) {
        printf("Timing error: %s\n", cudaGetErrorString(time_err));
    }

printf("Custom total time (ms): %.6f\n", custom_ms);
    const double cublas_us = cublas_ms * 1000.0 / repeats;

    cudaMemcpy(
    h_cublas.data(),
    d_C,
    bytes,
    cudaMemcpyDeviceToHost
    );

    // --------------------------------------------------------
    // Performance and forward error
    // --------------------------------------------------------
    const double flops = 2.0 * n * n * n * batch_size;

    const double custom_gflops = flops / (custom_us * 1.0e3);
    const double cublas_gflops = flops / (cublas_us * 1.0e3);

    const double custom_error = max_batch_forward_error(
                                    n, batch_size,
                                    h_custom.data(), h_ref.data());

    const double cublas_error = max_batch_forward_error(
                                    n, batch_size,
                                    h_cublas.data(), h_ref.data());

    std::printf("%-24s %12s %12s %20s\n",
                "Implementation",
                "Time (us)",
                "GFLOP/s",
                "Max batch fwd error");

    std::printf("-----------------------------------------------------------------------\n");

    std::printf("%-24s %12.3f %12.3f %20.6e\n",
                "Custom kernel",
                custom_us,
                custom_gflops,
                custom_error);

    std::printf("%-24s %12.3f %12.3f %20.6e\n",
                "cuBLAS Strided Batched",
                cublas_us,
                cublas_gflops,
                cublas_error);

    const double tol = 1e-6;
    if( double(custom_error) <= tol && double(cublas_error) <= tol ) {
        std::printf("\nTEST PASS\n");
        std::printf("Speedup (cuBLAS time / custom time): %.3fx\n", cublas_us / custom_us);
    }
    else {
        std::printf("\nTEST FAIL\n");
        std::printf("Speedup is not calculated\n");
    }

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cublasDestroy(handle);

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

    return 0;
}
