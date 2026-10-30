#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ============================================================
// Vérification des erreurs CUDA
// ============================================================
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// ============================================================
// Vérification des erreurs cuBLAS
// ============================================================
#define CUBLAS_CHECK(call)                                                 \
    do {                                                                   \
        cublasStatus_t status = (call);                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                             \
            fprintf(stderr, "cuBLAS error at %s:%d: status = %d\n",         \
                    __FILE__, __LINE__, (int)status);                      \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// ============================================================
// Custom batched GEMM
// C[b] = A[b] * B[b]
// Matrices N x N en column-major
// ============================================================
__global__ void batch_gemm_kernel(
    int n,
    int batch_size,
    const float* A,
    const float* B,
    float* C)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (row < n && col < n && b < batch_size)
    {
        size_t matrix_size = (size_t)n * n;
        size_t base = (size_t)b * matrix_size;

        float sum = 0.0f;

        for (int k = 0; k < n; ++k)
        {
            sum += A[base + (size_t)k * n + row]
                 * B[base + (size_t)col * n + k];
        }

        C[base + (size_t)col * n + row] = sum;
    }
}


// ============================================================
// CPU reference implementation
// ============================================================
void batched_gemm_cpu(
    int n,
    int batch_size,
    const float* A,
    const float* B,
    float* C)
{
    const size_t matrix_size = (size_t)n * n;

    for (int b = 0; b < batch_size; ++b)
    {
        const size_t base = (size_t)b * matrix_size;

        for (int col = 0; col < n; ++col)
        {
            for (int row = 0; row < n; ++row)
            {
                float sum = 0.0f;

                for (int k = 0; k < n; ++k)
                {
                    sum += A[base + (size_t)k * n + row]
                         * B[base + (size_t)col * n + k];
                }

                C[base + (size_t)col * n + row] = sum;
            }
        }
    }
}

// ============================================================
// Forward error for one matrix
// ||Cgpu - Cref||F / (N * ||Cref||F)
// ============================================================
double forward_error(
    int n,
    const float* result,
    const float* reference)
{
    double diff2 = 0.0;
    double ref2 = 0.0;

    const size_t size = (size_t)n * n;

    for (size_t i = 0; i < size; ++i)
    {
        double diff = (double)result[i] - (double)reference[i];

        diff2 += diff * diff;
        ref2 += (double)reference[i] * reference[i];
    }

    double diff_norm = std::sqrt(diff2);
    double ref_norm = std::sqrt(ref2);

    if (ref_norm > 0.0)
        return diff_norm / (n * ref_norm);

    return (diff_norm == 0.0) ? 0.0 : INFINITY;
}

// ============================================================
// Maximum forward error over the batch
// ============================================================
double max_batch_forward_error(
    int n,
    int batch_size,
    const float* result,
    const float* reference)
{
    const size_t matrix_size = (size_t)n * n;

    double max_error = 0.0;

    for (int b = 0; b < batch_size; ++b)
    {
        const size_t base = (size_t)b * matrix_size;

        double error = forward_error(
            n,
            result + base,
            reference + base
        );

        max_error = std::max(max_error, error);
    }

    return max_error;
}

// ============================================================
// Affichage de diagnostic des résultats
// ============================================================
void print_debug(
    int n,
    int batch_size,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& reference,
    const std::vector<float>& custom,
    const std::vector<float>& cublas)
{
    const size_t matrix_size = (size_t)n * n;
    const size_t last_batch = (size_t)(batch_size - 1) * matrix_size;

    printf("\n");
    printf("====================================================\n");
    printf("                 DEBUG INFORMATION\n");
    printf("====================================================\n");

    printf("\n--- INPUT DATA ---\n");
    printf("A[0] = %.9f\n", A[0]);
    printf("B[0] = %.9f\n", B[0]);

    printf("\n--- FIRST 10 RESULTS (BATCH 0) ---\n");
    printf("%-8s %-16s %-16s %-16s\n",
           "Index", "CPU", "Custom CUDA", "cuBLAS");

    for (int i = 0; i < std::min(10, n * n); ++i)
    {
        printf("%-8d %-16.9f %-16.9f %-16.9f\n",
               i,
               reference[i],
               custom[i],
               cublas[i]);
    }

    printf("\n--- FIRST RESULT OF LAST BATCH ---\n");
    printf("Batch index       : %d\n", batch_size - 1);
    printf("CPU reference     : %.9f\n", reference[last_batch]);
    printf("Custom CUDA       : %.9f\n", custom[last_batch]);
    printf("cuBLAS            : %.9f\n", cublas[last_batch]);

    printf("\n====================================================\n");
}

// ============================================================
// MAIN
// ============================================================
int main(int argc, char** argv)
{
    const int n = (argc > 1) ? std::atoi(argv[1]) : 16;
    const int batch_size = (argc > 2) ? std::atoi(argv[2]) : 1000;
    const int repeats = 100;

    if (n <= 0 || n > 32 || batch_size <= 0)
    {
        fprintf(stderr,
                "Usage: %s matrix_size(1..32) batch_size\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const size_t matrix_size = (size_t)n * n;
    const size_t total = matrix_size * batch_size;
    const size_t bytes = total * sizeof(float);

    // --------------------------------------------------------
    // 1. Allocation des tableaux CPU
    // --------------------------------------------------------
    std::vector<float> h_A(total);
    std::vector<float> h_B(total);
    std::vector<float> h_ref(total);
    std::vector<float> h_custom(total);
    std::vector<float> h_cublas(total);

    // --------------------------------------------------------
    // 2. Initialisation des matrices A et B
    // --------------------------------------------------------
    for (size_t i = 0; i < total; ++i)
    {
        h_A[i] = std::sin(0.013f * (float)(i + 1));
        h_B[i] = std::cos(0.017f * (float)(i + 1));
    }

    // Calcul de référence sur CPU
    batched_gemm_cpu(
        n, batch_size,
        h_A.data(),
        h_B.data(),
        h_ref.data()
    );

    // --------------------------------------------------------
    // 3. Allocation de la mémoire GPU
    // --------------------------------------------------------
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // --------------------------------------------------------
    // 4. Copie des matrices A et B vers le GPU
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), bytes, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), bytes, cudaMemcpyHostToDevice
    ));

    // --------------------------------------------------------
    // 5. Informations sur le GPU
    // --------------------------------------------------------
    int device = 0;
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU: %s (compute capability %d.%d)\n",
           prop.name, prop.major, prop.minor);

    printf("Matrix size: %d x %d, batch size: %d\n",
           n, n, batch_size);

    printf("FP32, column-major matrices, repeats: %d\n\n",
           repeats);

    // --------------------------------------------------------
    // 6. Création des événements CUDA
    // --------------------------------------------------------
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --------------------------------------------------------
    // 7. Configuration du kernel
    // --------------------------------------------------------
    dim3 block(16, 16);

    dim3 grid(
        (n + block.x - 1) / block.x,
        (n + block.y - 1) / block.y,
        batch_size
    );

    // --------------------------------------------------------
    // 8. Warmup du kernel CUDA
    // --------------------------------------------------------
    batch_gemm_kernel<<<grid, block>>>(
        n, batch_size, d_A, d_B, d_C
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // 9. Mesure du temps du kernel personnalisé
    // --------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start));

    for (int r = 0; r < repeats; ++r)
    {
        batch_gemm_kernel<<<grid, block>>>(
            n, batch_size, d_A, d_B, d_C
        );
    }

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float custom_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &custom_ms, start, stop
    ));

    const double custom_us =
        (double)custom_ms * 1000.0 / repeats;

    // --------------------------------------------------------
    // 10. Copie du résultat du kernel vers le CPU
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(
        h_custom.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // --------------------------------------------------------
    // 11. Création du handle cuBLAS
    // --------------------------------------------------------
    cublasHandle_t handle;

    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // --------------------------------------------------------
    // 12. Warmup cuBLAS
    // --------------------------------------------------------
    CUBLAS_CHECK(cublasSgemmStridedBatched(
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
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // 13. Mesure du temps cuBLAS
    // --------------------------------------------------------
    CUDA_CHECK(cudaEventRecord(start));

    for (int r = 0; r < repeats; ++r)
    {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
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
        ));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float cublas_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &cublas_ms, start, stop
    ));

    const double cublas_us =
        (double)cublas_ms * 1000.0 / repeats;

    // --------------------------------------------------------
    // 14. Copie du résultat cuBLAS vers le CPU
    // --------------------------------------------------------
    CUDA_CHECK(cudaMemcpy(
        h_cublas.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // --------------------------------------------------------
    // 15. AFFICHAGE DES DEBUGS
    // --------------------------------------------------------
    print_debug(
        n,
        batch_size,
        h_A,
        h_B,
        h_ref,
        h_custom,
        h_cublas
    );

    // --------------------------------------------------------
    // 16. Calcul des erreurs numériques
    // --------------------------------------------------------
    const double custom_error = max_batch_forward_error(
        n, batch_size,
        h_custom.data(),
        h_ref.data()
    );

    const double cublas_error = max_batch_forward_error(
        n, batch_size,
        h_cublas.data(),
        h_ref.data()
    );

    // --------------------------------------------------------
    // 17. Calcul des performances
    // --------------------------------------------------------
    const double flops =
        2.0 * n * n * n * batch_size;

    double custom_gflops = 0.0;
    double cublas_gflops = 0.0;

    if (custom_us > 0.0)
        custom_gflops = flops / (custom_us * 1.0e3);

    if (cublas_us > 0.0)
        cublas_gflops = flops / (cublas_us * 1.0e3);

    // --------------------------------------------------------
    // 18. Affichage des résultats
    // --------------------------------------------------------
    printf("\n");
    printf("%-24s %12s %12s %20s\n",
           "Implementation",
           "Time (us)",
           "GFLOP/s",
           "Max batch fwd error");

    printf("-----------------------------------------------------------------------\n");

    printf("%-24s %12.6f %12.6f %20.6e\n",
           "Custom kernel",
           custom_us,
           custom_gflops,
           custom_error);

    printf("%-24s %12.6f %12.6f %20.6e\n",
           "cuBLAS Strided Batched",
           cublas_us,
           cublas_gflops,
           cublas_error);

    printf("\nCustom total time (ms): %.9f\n", custom_ms);
    printf("cuBLAS total time (ms): %.9f\n", cublas_ms);

    // --------------------------------------------------------
    // 19. Vérification du test
    // --------------------------------------------------------
    const double tol = 1e-6;

    if (custom_error <= tol && cublas_error <= tol)
    {
        printf("\nTEST PASS\n");

        if (custom_us > 0.0)
        {
            printf("Speedup (cuBLAS time / custom time): %.3fx\n",
                   cublas_us / custom_us);
        }
    }
    else
    {
        printf("\nTEST FAIL\n");
        printf("Tolerance: %.1e\n", tol);
        printf("Custom error: %.12e\n", custom_error);
        printf("cuBLAS error: %.12e\n", cublas_error);
        printf("Speedup is not calculated\n");
    }

    // --------------------------------------------------------
    // 20. Libération des ressources
    // --------------------------------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
