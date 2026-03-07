#include <iostream>
#include <vector>
#include <stdexcept>

// ---- KERNEL 1: IMPLEMENTACIÓN SIMPLE (NAIVE) ----
// Cada hilo calcula un único elemento de la matriz de resultado C.
// Es ineficiente porque cada hilo lee una fila completa de A y una columna
// completa de B desde la lenta memoria global, con muchas lecturas repetidas
// entre los hilos del mismo bloque.
__global__ void matrixMulNaiveKernel(float *C, const float *A, const float *B, int M, int N, int K) {
    // Calcular la fila y columna global del elemento de C a calcular
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Asegurarse de no salirse de los límites de la matriz C
    if (row < M && col < N) {
        float sum = 0.0f;
        // Realizar el producto punto de la fila de A y la columna de B
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}


// ---- KERNEL 2: IMPLEMENTACIÓN OPTIMIZADA CON TILES (MEMORIA COMPARTIDA) ----
// Ancho del tile (submatriz). Debe ser un múltiplo de 32 para mejor rendimiento.
// Un bloque de hilos (TILE_WIDTH x TILE_WIDTH) calculará una submatriz de C.
const int TILE_WIDTH = 16;

__global__ void matrixMulTiledKernel(float *C, const float *A, const float *B, int M, int N, int K) {
    // 1. Declarar la memoria compartida (muy rápida, por bloque)
    // Cada bloque reserva espacio para un tile de A y un tile de B.
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Identificadores del hilo dentro del bloque (coordenadas locales en el tile)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Identificadores del bloque (coordenadas del tile en la matriz global)
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calcular la fila y columna global del elemento de C para este hilo
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // Registro privado para cada hilo para acumular el resultado
    float sum = 0.0f;

    // 2. Bucle sobre los tiles de A y B necesarios para calcular un tile de C
    for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++i) {
        // 3. Carga cooperativa desde la memoria global a la memoria compartida
        // Cada hilo del bloque carga un elemento de A y uno de B en sus respectivos tiles.
        if (row < M && (i * TILE_WIDTH + tx) < K) {
            As[ty][tx] = A[row * K + (i * TILE_WIDTH + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((i * TILE_WIDTH + ty) < K && col < N) {
            Bs[ty][tx] = B[(i * TILE_WIDTH + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Sincronizar todos los hilos del bloque para asegurar que los tiles
        // en memoria compartida (As y Bs) estén completamente cargados.
        __syncthreads();

        // 4. Calcular el producto punto desde la rápida memoria compartida
        for (int j = 0; j < TILE_WIDTH; ++j) {
            sum += As[ty][j] * Bs[j][tx];
        }

        // Sincronizar de nuevo antes de cargar el siguiente tile en el bucle
        __syncthreads();
    }

    // 5. Escribir el resultado final desde el registro a la memoria global
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// --- Función de ayuda en la CPU para verificar los resultados ---
void verifyResult(const float* C, const float* A, const float* B, int M, int N, int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            if (abs(C[row * N + col] - sum) > 1e-4) {
                throw std::runtime_error("¡Resultado incorrecto en la GPU!");
            }
        }
    }
    std::cout << "¡Verificacion exitosa! El resultado de la GPU es correcto." << std::endl;
}


int main() {
    // Dimensiones de las matrices (A: M x K, B: K x N, C: M x N)
    int M = 512;
    int N = 512;
    int K = 512;

    // Inicializar matrices en el Host (CPU)
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Punteros del Device (GPU)
    float *d_A, *d_B, *d_C;

    // Reservar memoria en la GPU
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Copiar datos del Host al Device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    // --- LANZAMIENTO DEL KERNEL ---
    // Usar una grid y bloques 2D para que coincida con la estructura de la matriz
    
    // Para el Kernel Optimizado con Tiles
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMulTiledKernel<<<numBlocks, threadsPerBlock>>>(d_C, d_A, d_B, M, N, K);

    /*
    // Para el Kernel Simple (Naive)
    dim3 threadsPerBlockNaive(16, 16);
    dim3 numBlocksNaive((N + threadsPerBlockNaive.x - 1) / threadsPerBlockNaive.x, (M + threadsPerBlockNaive.y - 1) / threadsPerBlockNaive.y);
    matrixMulNaiveKernel<<<numBlocksNaive, threadsPerBlockNaive>>>(d_C, d_A, d_B, M, N, K);
    */

    // Copiar resultados de vuelta al Host
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verificar los resultados
    try {
        verifyResult(h_C.data(), h_A.data(), h_B.data(), M, N, K);
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }
    
    // Liberar memoria de la GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
