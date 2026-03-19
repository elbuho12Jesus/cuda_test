cmake_minimum_required(VERSION 3.14)

# Habilitamos C++ y CUDA (Se corrigió la sintaxis)
project(MiProyecto LANGUAGES CXX CUDA)

# Creamos tu programa asegurando que termine en .cu
add_executable(MiPrograma main.cu)

# Le decimos al compilador dónde buscar la carpeta Eigen que descargaste manualmente
target_include_directories(MiPrograma PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})