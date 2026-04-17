#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel for tiled matrix multiplication using shared memory
// Each thread block computes a TILE_SIZE x TILE_SIZE submatrix of C
template<int TILE_SIZE>
__global__ void tiledMatMulKernel(const float *A, const float *B, float *C, int N) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global row and column for this thread
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // Accumulator for the dot product
    float sum = 0.0f;
    
    // Loop over tiles
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < N && a_col < N) {
            As[ty][tx] = A[row * N + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + ty;
        if (b_row < N && col < N) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to find optimal tile size
int getOptimalTileSize(int N) {
    // Preferred tile size is N/128 if N is divisible by 128
    // Otherwise, find the largest divisor <= 32 (max shared memory constraint)
    
    // Try N/128 first
    if (N >= 128 && N % 128 == 0) {
        int tile = N / 128;
        // Ensure tile size is reasonable (between 4 and 32)
        if (tile >= 4 && tile <= 32) {
            return tile;
        }
    }
    
    // Find largest divisor of N that is <= 32 and is a power of 2 or common size
    int candidates[] = {32, 16, 8, 4};
    for (int i = 0; i < 4; i++) {
        if (N % candidates[i] == 0) {
            return candidates[i];
        }
    }
    
    // If N is not divisible by common sizes, find largest divisor <= 32
    for (int tile = 32; tile >= 4; tile--) {
        if (N % tile == 0) {
            return tile;
        }
    }
    
    // Default fallback
    return 16;
}

// Host function to perform tiled matrix multiplication
void tiledMatMul(const float *h_A, const float *h_B, float *h_C, int N, int *tile_size_used = NULL) {
    // Calculate memory size
    size_t bytes = N * N * sizeof(float);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy input matrices from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Determine optimal tile size
    int TILE_SIZE = getOptimalTileSize(N);
    if (tile_size_used != NULL) {
        *tile_size_used = TILE_SIZE;
    }
    
    // Define thread block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, 
                 (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel with appropriate tile size
    if (TILE_SIZE == 32) {
        tiledMatMulKernel<32><<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    } else if (TILE_SIZE == 16) {
        tiledMatMulKernel<16><<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    } else if (TILE_SIZE == 8) {
        tiledMatMulKernel<8><<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    } else if (TILE_SIZE == 4) {
        tiledMatMulKernel<4><<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    } else {
        // For other tile sizes, use a generic approach or default to 16
        printf("Warning: Unsupported tile size %d, using 16 instead\n", TILE_SIZE);
        TILE_SIZE = 16;
        dim3 blockDim2(16, 16);
        dim3 gridDim2((N + 15) / 16, (N + 15) / 16);
        tiledMatMulKernel<16><<<gridDim2, blockDim2>>>(d_A, d_B, d_C, N);
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Helper function to initialize matrix with random values
void initMatrix(float *mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)rand() / RAND_MAX * 10.0f;
    }
}

// Helper function to check if file is .npy format
bool isNpyFile(const char *filename) {
    const char *ext = strrchr(filename, '.');
    return (ext != NULL && strcmp(ext, ".npy") == 0);
}

// Helper function to read .npy file (simplified for 2D float32 arrays)
bool readNpyMatrix(const char *filename, float **mat, int *N) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return false;
    }
    
    // Read magic string (6 bytes)
    char magic[6];
    if (fread(magic, 1, 6, fp) != 6) {
        printf("Error: Cannot read NPY magic string\n");
        fclose(fp);
        return false;
    }
    
    // Verify magic string
    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' || 
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        printf("Error: Invalid NPY file format\n");
        fclose(fp);
        return false;
    }
    
    // Read version (2 bytes)
    unsigned char version[2];
    if (fread(version, 1, 2, fp) != 2) {
        printf("Error: Cannot read NPY version\n");
        fclose(fp);
        return false;
    }
    
    // Read header length
    unsigned int header_len;
    if (version[0] == 1) {
        unsigned short len;
        if (fread(&len, 2, 1, fp) != 1) {
            printf("Error: Cannot read header length\n");
            fclose(fp);
            return false;
        }
        header_len = len;
    } else if (version[0] == 2 || version[0] == 3) {
        if (fread(&header_len, 4, 1, fp) != 1) {
            printf("Error: Cannot read header length\n");
            fclose(fp);
            return false;
        }
    } else {
        printf("Error: Unsupported NPY version %d.%d\n", version[0], version[1]);
        fclose(fp);
        return false;
    }
    
    // Read header (Python dict as string)
    char *header = (char*)malloc(header_len + 1);
    if (fread(header, 1, header_len, fp) != header_len) {
        printf("Error: Cannot read NPY header\n");
        free(header);
        fclose(fp);
        return false;
    }
    header[header_len] = '\0';
    
    // Parse shape from header
    char *shape_start = strstr(header, "'shape'");
    if (!shape_start) {
        shape_start = strstr(header, "\"shape\"");
    }
    if (!shape_start) {
        printf("Error: Cannot find shape in NPY header\n");
        free(header);
        fclose(fp);
        return false;
    }
    
    // Extract dimensions
    int dim1, dim2;
    char *paren = strchr(shape_start, '(');
    if (!paren || sscanf(paren, "(%d, %d)", &dim1, &dim2) != 2) {
        printf("Error: Cannot parse shape from NPY header\n");
        free(header);
        fclose(fp);
        return false;
    }
    
    // Check if matrix is square
    if (dim1 != dim2) {
        printf("Error: Matrix must be square (got %dx%d)\n", dim1, dim2);
        free(header);
        fclose(fp);
        return false;
    }
    
    *N = dim1;
    free(header);
    
    // Allocate memory for matrix
    *mat = (float*)malloc((*N) * (*N) * sizeof(float));
    if (*mat == NULL) {
        printf("Error: Memory allocation failed\n");
        fclose(fp);
        return false;
    }
    
    // Read data
    size_t elements = (*N) * (*N);
    if (fread(*mat, sizeof(float), elements, fp) != elements) {
        printf("Error: Cannot read matrix data from NPY file\n");
        free(*mat);
        fclose(fp);
        return false;
    }
    
    fclose(fp);
    return true;
}

// Helper function to write .npy file (2D float32 array)
bool writeNpyMatrix(const char *filename, const float *mat, int N) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename);
        return false;
    }
    
    // Write magic string
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    fwrite(magic, 1, 6, fp);
    
    // Write version (1.0)
    unsigned char version[] = {1, 0};
    fwrite(version, 1, 2, fp);
    
    // Create header
    char header[256];
    int header_len = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }",
        N, N);
    
    // Pad header to 64-byte alignment
    int padding = 64 - ((6 + 2 + 2 + header_len) % 64);
    if (padding == 64) padding = 0;
    for (int i = 0; i < padding; i++) {
        header[header_len++] = ' ';
    }
    header[header_len++] = '\n';
    
    // Write header length
    unsigned short hlen = header_len;
    fwrite(&hlen, 2, 1, fp);
    
    // Write header
    fwrite(header, 1, header_len, fp);
    
    // Write data
    fwrite(mat, sizeof(float), N * N, fp);
    
    fclose(fp);
    printf("Matrix written to %s (NumPy format)\n", filename);
    return true;
}

// Helper function to read matrix from file (auto-detect format)
bool readMatrixFromFile(const char *filename, float **mat, int *N) {
    if (isNpyFile(filename)) {
        return readNpyMatrix(filename, mat, N);
    }
    
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Cannot open file %s\n", filename);
        return false;
    }
    
    if (fscanf(fp, "%d", N) != 1) {
        printf("Error: Cannot read matrix size from %s\n", filename);
        fclose(fp);
        return false;
    }
    
    *mat = (float*)malloc((*N) * (*N) * sizeof(float));
    if (*mat == NULL) {
        printf("Error: Memory allocation failed\n");
        fclose(fp);
        return false;
    }
    
    for (int i = 0; i < (*N) * (*N); i++) {
        if (fscanf(fp, "%f", &(*mat)[i]) != 1) {
            printf("Error: Cannot read matrix element %d from %s\n", i, filename);
            free(*mat);
            fclose(fp);
            return false;
        }
    }
    
    fclose(fp);
    return true;
}

// Helper function to write matrix to file (auto-detect format from extension)
bool writeMatrixToFile(const char *filename, const float *mat, int N) {
    if (isNpyFile(filename)) {
        return writeNpyMatrix(filename, mat, N);
    }
    
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open file %s for writing\n", filename);
        return false;
    }
    
    fprintf(fp, "%d\n", N);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fp, "%f", mat[i * N + j]);
            if (j < N - 1) fprintf(fp, " ");
        }
        fprintf(fp, "\n");
    }
    
    fclose(fp);
    printf("Matrix written to %s\n", filename);
    return true;
}

// Helper function to print matrix (for small matrices)
void printMatrix(const char *name, const float *mat, int N, int max_display = 8) {
    printf("\nMatrix %s (%dx%d):\n", name, N, N);
    int display_size = (N < max_display) ? N : max_display;
    
    for (int i = 0; i < display_size; i++) {
        for (int j = 0; j < display_size; j++) {
            printf("%8.2f ", mat[i * N + j]);
        }
        if (N > max_display) printf("...");
        printf("\n");
    }
    if (N > max_display) {
        printf("...\n");
    }
}

// Helper function to verify results (CPU matrix multiplication)
void cpuMatMul(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Helper function to check if two matrices are approximately equal
bool verifyResults(const float *C_gpu, const float *C_cpu, int N, float tolerance = 1e-3) {
    for (int i = 0; i < N * N; i++) {
        if (fabs(C_gpu[i] - C_cpu[i]) > tolerance) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", 
                   i, C_gpu[i], C_cpu[i]);
            return false;
        }
    }
    return true;
}

// Main function demonstrating usage
int main(int argc, char **argv) {
    float *h_A = NULL, *h_B = NULL, *h_C = NULL, *h_C_ref = NULL;
    int N = 0;
    bool from_files = false;
    
    // Parse command line arguments
    if (argc >= 3) {
        printf("Reading matrices from files...\n");
        
        int N_A, N_B;
        if (!readMatrixFromFile(argv[1], &h_A, &N_A)) {
            return 1;
        }
        if (!readMatrixFromFile(argv[2], &h_B, &N_B)) {
            free(h_A);
            return 1;
        }
        
        if (N_A != N_B) {
            printf("Error: Matrix dimensions don't match (A: %dx%d, B: %dx%d)\n", 
                   N_A, N_A, N_B, N_B);
            free(h_A);
            free(h_B);
            return 1;
        }
        
        N = N_A;
        from_files = true;
        
        size_t bytes = N * N * sizeof(float);
        h_C = (float*)malloc(bytes);
        h_C_ref = (float*)malloc(bytes);
        
        printf("Loaded matrices A and B (%dx%d)\n", N, N);
        
    } else if (argc == 2) {
        N = atoi(argv[1]);
        
        if (N <= 0) {
            printf("Error: Invalid matrix size %d\n", N);
            return 1;
        }
        
        size_t bytes = N * N * sizeof(float);
        h_A = (float*)malloc(bytes);
        h_B = (float*)malloc(bytes);
        h_C = (float*)malloc(bytes);
        h_C_ref = (float*)malloc(bytes);
        
        srand(42);
        initMatrix(h_A, N);
        initMatrix(h_B, N);
        
        printf("Generated random matrices (%dx%d)\n", N, N);
        
    } else {
        N = 128;  // Default to 128 for tiled version
        
        size_t bytes = N * N * sizeof(float);
        h_A = (float*)malloc(bytes);
        h_B = (float*)malloc(bytes);
        h_C = (float*)malloc(bytes);
        h_C_ref = (float*)malloc(bytes);
        
        srand(42);
        initMatrix(h_A, N);
        initMatrix(h_B, N);
        
        printf("Generated random matrices (default: %dx%d)\n", N, N);
    }
    
    printf("\nTiled Matrix Multiplication: %d x %d\n", N, N);
    
    // Display input matrices (if small enough)
    if (N <= 8) {
        printMatrix("A", h_A, N);
        printMatrix("B", h_B, N);
    }
    
    // Perform GPU matrix multiplication with tiling
    printf("\nRunning GPU tiled matrix multiplication...\n");
    int tile_size;
    tiledMatMul(h_A, h_B, h_C, N, &tile_size);
    printf("Used tile size: %dx%d\n", tile_size, tile_size);
    printf("Number of tiles: %dx%d = %d total tiles\n", 
           (N + tile_size - 1) / tile_size, 
           (N + tile_size - 1) / tile_size,
           ((N + tile_size - 1) / tile_size) * ((N + tile_size - 1) / tile_size));
    
    // Calculate DRAM traffic reduction
    float reduction_factor = (float)tile_size;
    printf("DRAM traffic reduction: %.1fx compared to naive implementation\n", reduction_factor);
    
    // Perform CPU matrix multiplication for verification
    printf("\nRunning CPU matrix multiplication for verification...\n");
    cpuMatMul(h_A, h_B, h_C_ref, N);
    
    // Verify results
    if (verifyResults(h_C, h_C_ref, N)) {
        printf("✓ Verification PASSED: GPU and CPU results match!\n");
    } else {
        printf("✗ Verification FAILED: GPU and CPU results differ!\n");
    }
    
    // Display output matrix (if small enough)
    if (N <= 8) {
        printMatrix("C (GPU)", h_C, N);
    } else {
        printf("\nSample output (first 4x4 block of matrix C):\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%8.2f ", h_C[i * N + j]);
            }
            printf("\n");
        }
    }
    
    // Save output to file if input was from files
    if (from_files) {
        const char *output_file = (argc >= 4) ? argv[3] : "output_C.npy";
        writeMatrixToFile(output_file, h_C, N);
    }
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    return 0;
}
