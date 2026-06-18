/*
 * Instituto Politecnico Nacional
 * EScuela Superior de Computo
 * Computo Paralelo 
 * Práctica 8. Programación de cómputo paralelo con CUDA 
 * integrantes del equipo 
 *     - Bustillos Cruz Jonatan
 *     - Delgado Lucero Cristian Isaac
 *     - Frem Cortés José Angel
 *     - Luna Gonzales Gabriel Alexis
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#define Filas 10
#define Columnas 10

// Función auxiliar para imprimir la hora con milisegundos y una línea divisoria
void print_time_info(const char *prefix, const char *label) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm* tm_info = localtime(&tv.tv_sec);
    char buffer[26];
    strftime(buffer, 26, "%H:%M:%S", tm_info);
    printf("=========================================================\n");
    printf("%s [Hora: %s.%03d] %s\n", prefix, buffer, (int)(tv.tv_usec / 1000), label);
    printf("=========================================================\n\n");
}

// Imprime una matriz de tamaño 10x10 como una tabla limpia con bordes y alineación
void imprimir_matriz(const char *prefix, int mat[Filas][Columnas], const char *nombre) {
    printf("%s Matriz %s (%dx%d):\n", prefix, nombre, Filas, Columnas);
    
    // Línea superior de la tabla
    printf("%s +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+\n", prefix);
    for (int i = 0; i < Filas; i++) {
        printf("%s |", prefix);
        for (int j = 0; j < Columnas; j++) {
            printf(" %3d |", mat[i][j]);
        }
        printf("\n");
        printf("%s +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+\n", prefix);
    }
    printf("\n");
}

int main()
{
    // Medir tiempo de ejecución del programa de principio a fin
    struct timeval start_total, end_total;
    gettimeofday(&start_total, NULL);

    int A[Filas][Columnas];
    int B[Filas][Columnas];
    int C[Filas][Columnas] = {0};
    int i, j, k;

    const char *p_serial = "[PROCESO SERIAL C]";

    // Mostrar hora en la que se inicia
    print_time_info(p_serial, "Iniciando programa C (Esperando datos de CUDA)");

    // Esperar y copiar los números generados aleatoriamente por el programa CUDA
    FILE *f = NULL;
    while ((f = fopen("shared_matrices.txt", "r")) == NULL) {
        usleep(1000); // Esperar 1 ms antes de volver a intentar
    }

    // Copiar datos de la matriz A
    for (i = 0; i < Filas; i++) {
        for (j = 0; j < Columnas; j++) {
            if (fscanf(f, "%d", &A[i][j]) != 1) {
                fprintf(stderr, "%s Error al leer matriz A del archivo compartido.\n", p_serial);
                fclose(f);
                return 1;
            }
        }
    }

    // Copiar datos de la matriz B
    for (i = 0; i < Filas; i++) {
        for (j = 0; j < Columnas; j++) {
            if (fscanf(f, "%d", &B[i][j]) != 1) {
                fprintf(stderr, "%s Error al leer matriz B del archivo compartido.\n", p_serial);
                fclose(f);
                return 1;
            }
        }
    }
    fclose(f);

    // Eliminar el archivo compartido después de leerlo para dejar el directorio limpio
    remove("shared_matrices.txt");

    // Mostrar el tiempo del contador iniciado justo después de que se copiaron los datos del cuda
    print_time_info(p_serial, "Datos de CUDA copiados, iniciando cronómetro y cargando matrices");

    // Mostrar las matrices copiadas
    imprimir_matriz(p_serial, A, "A (Copiada de CUDA)");
    imprimir_matriz(p_serial, B, "B (Copiada de CUDA)");

    // Calcular multiplicación de matrices de forma serial
    for (i = 0; i < Filas; i++)
    {
        for (k = 0; k < Columnas; k++)
        {
            for (j = 0; j < Filas; j++)
            {
                C[i][k] = C[i][k] + A[i][j] * B[j][k];
            }
        }
    }

    // Mostrar el resultado de la multiplicación
    print_time_info(p_serial, "Cálculo completado. Mostrando Matriz C (Resultado Serial)");
    imprimir_matriz(p_serial, C, "C (Resultado Serial)");

    // Mostrar la hora en la que termina el programa de C
    print_time_info(p_serial, "Finalizando programa C");

    // Calcular tiempo total transcurrido
    gettimeofday(&end_total, NULL);
    double elapsed = (end_total.tv_sec - start_total.tv_sec) + 
                     (end_total.tv_usec - start_total.tv_usec) / 1000000.0;
                     
    printf("=========================================================\n");
    printf("%s Tiempo de ejecución total del programa C (inicio a fin): %.6f segundos\n", p_serial, elapsed);
    printf("=========================================================\n");

    return 0;
}
