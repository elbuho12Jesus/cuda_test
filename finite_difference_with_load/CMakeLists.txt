# Versión mínima de CMake.
cmake_minimum_required(VERSION 3.12)

# Habilita C++ y CUDA como lenguajes.
project(finiteDifference_2D LANGUAGES CXX CUDA)

# Añade el ejecutable (asegúrate de que tu fichero se llame matrix_mul.cu)
add_executable(${PROJECT_NAME} finite_difference_2D.cu)


target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# Enlaza las librerías de CUDA.
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
# Establece las propiedades de C++ y CUDA.
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/resources"
    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/resources"
)