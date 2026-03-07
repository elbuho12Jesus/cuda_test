# Versión mínima de CMake.
cmake_minimum_required(VERSION 3.12)

# Habilita C++ y CUDA como lenguajes.
project(MatrixMultiplication LANGUAGES CXX CUDA)

# Añade el ejecutable (asegúrate de que tu fichero se llame matrix_mul.cu)
add_executable(matrix_mul matrix_mul.cu)

# Enlaza las librerías de CUDA usando la variable "clásica".
target_link_libraries(matrix_mul PRIVATE ${CUDA_LIBRARIES})

# Establece las propiedades de C++ y CUDA.
set_target_properties(matrix_mul PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
