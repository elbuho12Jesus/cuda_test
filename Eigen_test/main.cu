#include <iostream>
// Así se incluye la librería en tu código
#include <Eigen/Dense>

int main() {
    // Declaramos una matriz de 2x2 de tipo double (decimales)
    Eigen::MatrixXd matriz(2,2);
    
    // Llenamos la matriz con valores
    matriz(0,0) = 3;
    matriz(1,0) = 2.5;
    matriz(0,1) = -1;
    matriz(1,1) = matriz(1,0) + matriz(0,1);
    
    // Eigen permite imprimir matrices directamente en la consola
    std::cout << "Mi primera matriz con Eigen es:\n" << matriz << std::endl;
    
    return 0;
}