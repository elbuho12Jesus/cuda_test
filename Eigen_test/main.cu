#include <iostream>
#include <Eigen/Dense>

// 1. Definimos el "Kernel" de CUDA que se ejecutará en la tarjeta gráfica
__global__ void multiplicarMatricesCUDA(const float* A, const float* B, float* C, int N) {
    // Calculamos la fila y la columna que este hilo (thread) específico va a procesar
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int columna = blockIdx.x * blockDim.x + threadIdx.x;

    // Nos aseguramos de no salirnos de los límites de la matriz
    if (fila < N && columna < N) {
        float suma = 0.0f;
        // Realizamos el producto punto de la fila de A y la columna de B
        for (int k = 0; k < N; ++k) {
            suma += A[fila * N + k] * B[k * N + columna];
        }
        // Guardamos el resultado en la matriz C
        C[fila * N + columna] = suma;
    }
}

int main() {
    int N = 2; // Tamaño de las matrices (2x2)
    size_t bytes = N * N * sizeof(float);

    // CRÍTICO: Eigen usa formato de columnas por defecto. 
    // Para que funcione fácil con CUDA (que usa formato de filas), forzamos RowMajor.
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrizEigen;

    // 2. Creamos las matrices en el procesador (CPU / Host)
    MatrizEigen h_A(N, N);
    MatrizEigen h_B(N, N);
    MatrizEigen h_C(N, N);

    // Llenamos las matrices con datos
    h_A << 1, 2, 
           3, 4;
           
    h_B << 5, 6, 
           7, 8;

    // 3. Reservamos memoria en la tarjeta gráfica (GPU / Device)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 4. Copiamos los datos de Eigen (CPU) hacia CUDA (GPU) usando .data()
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // 5. Configuramos cuántos "hilos" trabajarán al mismo tiempo y lanzamos el Kernel
    dim3 hilosPorBloque(16, 16);
    dim3 bloques((N + hilosPorBloque.x - 1) / hilosPorBloque.x, 
                 (N + hilosPorBloque.y - 1) / hilosPorBloque.y);
                 
    multiplicarMatricesCUDA<<<bloques, hilosPorBloque>>>(d_A, d_B, d_C, N);

    // 6. Traemos el resultado de regreso desde la GPU hacia la matriz Eigen en la CPU
    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    // Imprimimos el resultado formateado gracias a Eigen
    std::cout << "El resultado de la multiplicación es:\n" << h_C << std::endl;

    // 7. Liberamos la memoria de la tarjeta gráfica
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);

    return 0;
}