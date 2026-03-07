# Versión mínima de CMake.
cmake_minimum_required(VERSION 3.12)

# Habilita C++ y CUDA como lenguajes desde el principio.
# Esto es CRUCIAL para resolver el error "Cannot determine link language".
project(SumaVectores LANGUAGES CXX CUDA)

# Añade el ejecutable.
add_executable(suma_vectores vector_add.cu)

# Enlaza las librerías de CUDA usando la variable "clásica".
# Cuando habilitas el lenguaje CUDA, CMake define automáticamente la variable
# ${CUDA_LIBRARIES}. Esta contiene la ruta a libcudart.so.
# Es el método más robusto y compatible.
target_link_libraries(suma_vectores PRIVATE ${CUDA_LIBRARIES})

# Establece las propiedades de C++ y CUDA.
set_target_properties(suma_vectores PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)