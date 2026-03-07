# Versión mínima de CMake.
cmake_minimum_required(VERSION 3.12)

# Habilita C++ y CUDA como lenguajes.
project(finiteDifference_2D LANGUAGES CXX CUDA)

# Añade el ejecutable (asegúrate de que tu fichero se llame matrix_mul.cu)
add_executable(finite_difference_2D finite_difference_2D.cu)

# Enlaza las librerías de CUDA usando la variable "clásica".
target_link_libraries(finite_difference_2D PRIVATE ${CUDA_LIBRARIES})

# Establece las propiedades de C++ y CUDA.
set_target_properties(finite_difference_2D PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)