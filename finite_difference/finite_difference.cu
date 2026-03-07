#include <iostream>
#include <vector>
#include <cmath>

// Kernel para aplicar el stencil de 5 puntos (promedio de vecinos)
// Cada hilo calcula un punto de la grid de salida.
__global__ void stencil_2d_kernel(float* grid_out, const float* grid_in, int width, int height) {
    // Calcular la coordenada (x, y) global para este hilo
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Los hilos en los bordes no hacen nada, ya que no tienen todos los vecinos.
    // (Esta es la condición de contorno más simple, llamada "Dirichlet con valor 0")
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Coordenadas del punto actual y sus vecinos en la grid de entrada
        int current_pos = y * width + x;
        int north_pos = (y - 1) * width + x;
        int south_pos = (y + 1) * width + x;
        int west_pos = y * width + (x - 1);
        int east_pos = y * width + (x + 1);

        // Leer los valores de los vecinos desde la memoria global
        float north_val = grid_in[north_pos];
        float south_val = grid_in[south_pos];
        float west_val = grid_in[west_pos];
        float east_val = grid_in[east_pos];

        // Calcular el promedio y escribirlo en la grid de salida
        grid_out[current_pos] = (north_val + south_val + west_val + east_val) * 0.25f;
    }
}

// Función en el Host para inicializar la grid con un patrón
void initialize_grid(std::vector<float>& h_grid, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Poner un "punto caliente" en el centro
            if (x > width/2 - 10 && x < width/2 + 10 && y > height/2 - 10 && y < height/2 + 10) {
                 h_grid[y * width + x] = 100.0f;
            } else {
                 h_grid[y * width + x] = 0.0f;
            }
        }
    }
}

// Función en el Host para imprimir una sección de la grid
void print_grid_section(const std::vector<float>& h_grid, int width, int height) {
    std::cout << "--- Mostrando esquina superior izquierda de la grid ---" << std::endl;
    for (int y = 506; y <520; ++y) {
        for (int x = 506; x < 520; ++x) {
            printf("%5.1f ", h_grid[y * width + x]);
        }
        std::cout << std::endl;
    }
    std::cout << "---------------------------------------------------" << std::endl;
}

int main() {
    // Dimensiones de la grid
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const size_t size = WIDTH * HEIGHT * sizeof(float);
    const int NUM_ITERATIONS = 100;

    // Reservar memoria en el Host (CPU)
    std::vector<float> h_grid(WIDTH * HEIGHT);
    
    // Inicializar la grid en el Host
    initialize_grid(h_grid, WIDTH, HEIGHT);
    
    std::cout << "Estado inicial de la grid:" << std::endl;
    print_grid_section(h_grid, WIDTH, HEIGHT); // Descomentar para ver la grid    
    // Reservar memoria en el Device (GPU) para dos grids
    float *d_grid_in, *d_grid_out;
    cudaMalloc(&d_grid_in, size);
    cudaMalloc(&d_grid_out, size);

    // Copiar la grid inicial del Host a d_grid_in
    cudaMemcpy(d_grid_in, h_grid.data(), size, cudaMemcpyHostToDevice);
    // Inicializar d_grid_out a ceros (o copiar h_grid también)
    cudaMemcpy(d_grid_out, h_grid.data(), size, cudaMemcpyHostToDevice);

    // Configuración de la ejecución del Kernel (Grid 2D y Bloques 2D)
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "Realizando " << NUM_ITERATIONS << " iteraciones de diferencias finitas..." << std::endl;

    // Bucle de simulación
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // Lanzar el kernel
        stencil_2d_kernel<<<numBlocks, threadsPerBlock>>>(d_grid_out, d_grid_in, WIDTH, HEIGHT);

        // --- Intercambio de Punteros (Ping-Pong) ---
        // El resultado (out) de esta iteración se convierte en la entrada (in) de la siguiente.
        // Esto evita tener que hacer un cudaMemcpy en cada paso, lo cual sería muy lento.
        float* temp = d_grid_in;
        d_grid_in = d_grid_out;
        d_grid_out = temp;
    }

    // Después del bucle, el resultado final está en d_grid_in (debido al último intercambio)
    // Copiar el resultado final de vuelta al Host
    cudaMemcpy(h_grid.data(), d_grid_in, size, cudaMemcpyDeviceToHost);

    std::cout << "Simulacion completada." << std::endl;
    std::cout << "Estado final de la grid:" << std::endl;
    print_grid_section(h_grid, WIDTH, HEIGHT);

    // Liberar memoria de la GPU
    cudaFree(d_grid_in);
    cudaFree(d_grid_out);

    return 0;
}
