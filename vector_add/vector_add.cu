
#include <iostream>
#include <vector>

// --- INICIO DEL CÓDIGO DE LA GPU (DEVICE) ---

__global__ void sumarVectores(float *d_c, const float *d_a, const float *d_b, int n) {
    // ---- USO DE REGISTROS ----
    // 'i' es una variable local del kernel. El compilador de CUDA la almacenará en un
    // REGISTRO, que es la memoria más rápida y es privada para cada hilo.
    // Las variables intrínsecas 'blockIdx.x', 'blockDim.x', y 'threadIdx.x' también
    // son cargadas en registros para un acceso ultra-rápido.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        // ---- ACCESO A MEMORIA GLOBAL ----
        // Aquí es donde ocurre la mayor parte del trabajo de memoria.
        // 1. Leer d_a[i]: El hilo accede a la MEMORIA GLOBAL de la GPU para leer un float.
        // 2. Leer d_b[i]: El hilo accede a la MEMORIA GLOBAL de la GPU para leer otro float.
        // 3. Escribir d_c[i]: El hilo escribe el resultado de la suma en la MEMORIA GLOBAL.
        //
        // d_a, d_b, y d_c son punteros a la memoria global. Cada acceso es relativamente lento.
        // Sin embargo, como miles de hilos hacen esto simultáneamente, se logra un gran rendimiento.
        d_c[i] = d_a[i] + d_b[i];
    }
}

// --- FIN DEL CÓDIGO DE LA GPU ---


// --- INICIO DEL CÓDIGO DEL HOST (CPU) ---
int main() {
    int n = 1024;
    size_t size = n * sizeof(float);

    // ---- USO DE MEMORIA DEL HOST (RAM) ----
    // h_a, h_b, y h_c son vectores estándar de C++ que residen en la memoria RAM de la CPU.
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Punteros que apuntarán a direcciones en la memoria de la GPU.
    float *d_a, *d_b, *d_c;

    // ---- RESERVA DE MEMORIA GLOBAL EN LA GPU ----
    // cudaMalloc reserva 'size' bytes en la MEMORIA GLOBAL de la GPU.
    // d_a, d_b, y d_c ahora contienen las direcciones de inicio de estos bloques de memoria.
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // ---- TRANSFERENCIA DE DATOS: HOST -> DEVICE ----
    // cudaMemcpy copia los datos desde la RAM del Host (h_a, h_b) a la MEMORIA GLOBAL del Device (d_a, d_b).
    // Esta es una operación explícita a través del bus PCIe, crucial para la ejecución.
    cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

    // Configuración para el lanzamiento del Kernel
    int hilosPorBloque = 256;
    int bloquesPorGrid = (n + hilosPorBloque - 1) / hilosPorBloque;

    // ---- LANZAMIENTO DEL KERNEL ----
    // Se invoca la función 'sumarVectores' para que se ejecute en la GPU.
    // Los punteros d_c, d_a, d_b se pasan al kernel para que sepa dónde
    // encontrar los datos en la MEMORIA GLOBAL.
    sumarVectores<<<bloquesPorGrid, hilosPorBloque>>>(d_c, d_a, d_b, n);

    // ---- TRANSFERENCIA DE DATOS: DEVICE -> HOST ----
    // Después de que la GPU ha terminado, cudaMemcpy copia el vector de resultados (d_c)
    // desde la MEMORIA GLOBAL de la GPU de vuelta a la RAM del Host (h_c).
    cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

    // Verificación de los resultados en la CPU
    for (int i = 0; i < 10; ++i) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }

    // ---- LIBERACIÓN DE MEMORIA GLOBAL EN LA GPU ----
    // Es fundamental liberar la memoria de la GPU cuando ya no se necesita.
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}


