#include <iostream>
#include <vector>
#include <stdexcept>

// --- Integración de stb_image ---
// La implementación de AMBAS librerías debe definirse UNA SOLA VEZ
// y en el ámbito global (aquí arriba).
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
// ---------------------------------

const int TILE_WIDTH = 16;

__global__ void stencil_2d_kernel(float* grid_out, const float* grid_in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int current_pos = y * width + x;
        int north_pos = (y - 1) * width + x;
        int south_pos = (y + 1) * width + x;
        int west_pos = y * width + (x - 1);
        int east_pos = y * width + (x + 1);

        float north_val = grid_in[north_pos];
        float south_val = grid_in[south_pos];
        float west_val = grid_in[west_pos];
        float east_val = grid_in[east_pos];

        grid_out[current_pos] = (north_val + south_val + west_val + east_val) * 0.25f;
    }
}

int main() {
    // --- Carga de la Imagen ---
    int width, height, channels;
    const char* filename = "resources/mask_sr.png"; // Ruta relativa

    unsigned char *image_data = stbi_load(filename, &width, &height, &channels, 1);
    if (image_data == nullptr) {
        std::cerr << "Error: No se pudo cargar la imagen " << filename << std::endl;
        // Imprime el error específico de stb para más pistas
        std::cerr << "Motivo: " << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::cout << "Imagen cargada: " << width << "x" << height << " pixeles." << std::endl;
    // ---------------------------

    const size_t size_bytes = width * height * sizeof(float);
    const int NUM_ITERATIONS = 100;

    std::vector<float> h_grid(width * height);
    for (int i = 0; i < width * height; ++i) {
        h_grid[i] = static_cast<float>(image_data[i]);
    }
    stbi_image_free(image_data);

    // --- Lógica de CUDA ---
    float *d_grid_in, *d_grid_out;
    cudaMalloc(&d_grid_in, size_bytes);
    cudaMalloc(&d_grid_out, size_bytes);

    cudaMemcpy(d_grid_in, h_grid.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_out, h_grid.data(), size_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "Aplicando " << NUM_ITERATIONS << " iteraciones de suavizado..." << std::endl;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        stencil_2d_kernel<<<numBlocks, threadsPerBlock>>>(d_grid_out, d_grid_in, width, height);
        float* temp = d_grid_in;
        d_grid_in = d_grid_out;
        d_grid_out = temp;
    }

    cudaMemcpy(h_grid.data(), d_grid_in, size_bytes, cudaMemcpyDeviceToHost);
    std::cout << "Simulacion completada." << std::endl;

    // --- Guardar la imagen resultado ---
    std::vector<unsigned char> output_image_data(width * height);
    for(int i = 0; i < width * height; ++i) {
        float val = h_grid[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        output_image_data[i] = static_cast<unsigned char>(val);
    }
    
    stbi_write_png("output.png", width, height, 1, output_image_data.data(), width * sizeof(unsigned char));
    std::cout << "Imagen resultado guardada como output.png" << std::endl;
    // ------------------------------------

    cudaFree(d_grid_in);
    cudaFree(d_grid_out);

    return 0;
}