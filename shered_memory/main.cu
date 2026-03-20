#include <iostream>
#include <vector>

// Definimos el tamaño del bloque. 1024 es el máximo permitido por bloque en CUDA moderno.
#define TAMANO_BLOQUE 1024

// ---------------------------------------------------------
// 1. EL KERNEL DE LA GPU (Se ejecuta en la tarjeta gráfica)
// ---------------------------------------------------------
__global__ void sumarConMemoriaCompartida(int *entrada_global, int *salidas_parciales, int N) {
    // Declaramos la Memoria Compartida. Su tamaño debe coincidir con los hilos del bloque.
    __shared__ int pizarra[TAMANO_BLOQUE];

    int hilo_local = threadIdx.x; // ID dentro del bloque (0 a 1023)
    int id_global = blockIdx.x * blockDim.x + threadIdx.x; // ID en toda la cuadrícula

    // Cada hilo carga un número de la Memoria Global a la Memoria Compartida.
    // Si el id_global se sale del tamaño del arreglo, ponemos un 0 para no afectar la suma.
    if (id_global < N) {
        pizarra[hilo_local] = entrada_global[id_global];
    } else {
        pizarra[hilo_local] = 0;
    }

    // Sincronizamos: Nadie avanza hasta que los 1024 números estén en la pizarra
    __syncthreads();

    // Empieza el "Torneo" de reducción
    for (int salto = blockDim.x / 2; salto > 0; salto /= 2) {
        if (hilo_local < salto) {
            pizarra[hilo_local] += pizarra[hilo_local + salto];
        }
        // Sincronizamos después de cada ronda del torneo
        __syncthreads();
    }

    // El hilo capitán (0) de cada bloque escribe el resultado del bloque en la Memoria Global
    if (hilo_local == 0) {
        salidas_parciales[blockIdx.x] = pizarra[0];
    }
}

// ---------------------------------------------------------
// 2. EL PROGRAMA PRINCIPAL (Se ejecuta en el procesador)
// ---------------------------------------------------------
int main() {
    // Vamos a sumar 1 millón de números (1024 * 1024)
    int N = 1048576;
    size_t bytes = N * sizeof(int);

    std::cout << "Iniciando suma de " << N << " elementos..." << std::endl;

    // Calculamos cuántos bloques necesitamos en la cuadrícula
    // N / 1024 = 1024 bloques en total.
    int numero_de_bloques = (N + TAMANO_BLOQUE - 1) / TAMANO_BLOQUE;
    size_t bytes_parciales = numero_de_bloques * sizeof(int);

    // Creamos la memoria en la CPU (Host)
    std::vector<int> h_entrada(N, 1); // Un arreglo de 1 millón de "1"s. La suma debe dar 1,048,576.
    std::vector<int> h_salidas_parciales(numero_de_bloques, 0);

    // Punteros para la memoria de la GPU (Device)
    int *d_entrada, *d_salidas_parciales;

    // Reservamos memoria en la tarjeta gráfica
    cudaMalloc(&d_entrada, bytes);
    cudaMalloc(&d_salidas_parciales, bytes_parciales);

    // Copiamos los datos de la CPU a la GPU
    cudaMemcpy(d_entrada, h_entrada.data(), bytes, cudaMemcpyHostToDevice);

    // Lanzamos el Kernel a la tarjeta gráfica
    sumarConMemoriaCompartida<<<numero_de_bloques, TAMANO_BLOQUE>>>(d_entrada, d_salidas_parciales, N);

    // Traemos los resultados parciales de regreso a la CPU
    cudaMemcpy(h_salidas_parciales.data(), d_salidas_parciales, bytes_parciales, cudaMemcpyDeviceToHost);

    // ---------------------------------------------------------
    // 3. LA SUMA FINAL EN LA CPU
    // ---------------------------------------------------------
    // La GPU nos devolvió 1024 sumas (una por cada bloque). 
    // Es tan poco trabajo que es más rápido que la CPU termine de sumarlos en lugar de lanzar la GPU de nuevo.
    long long suma_total = 0;
    for (int i = 0; i < numero_de_bloques; i++) {
        suma_total += h_salidas_parciales[i];
    }

    std::cout << "Resultado final de la suma: " << suma_total << std::endl;
    if (suma_total == N) {
        std::cout << "¡Exito! La matematica es correcta." << std::endl;
    } else {
        std::cout << "Error en el calculo." << std::endl;
    }

    // Limpiamos la memoria de la tarjeta gráfica
    cudaFree(d_entrada);
    cudaFree(d_salidas_parciales);

    return 0;
}