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
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 10

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

// Imprime una matriz de tamaño NxN como una tabla limpia con bordes y alineación
void imprimir_matriz(const char *prefix, int mat[N][N], const char *nombre) {
    printf("%s Matriz %s (%dx%d):\n", prefix, nombre, N, N);
    
    // Línea superior de la tabla
    printf("%s +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+\n", prefix);
    for (int i = 0; i < N; i++) {
        printf("%s |", prefix);
        for (int j = 0; j < N; j++) {
            printf(" %3d |", mat[i][j]);
        }
        printf("\n");
        printf("%s +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+\n", prefix);
    }
    printf("\n");
}

// Kernel de CUDA para multiplicar matrices utilizando hilos y bloques
__global__ void MultiplicarMatricesGPU(int *A, int *B, int *C, int tam)
{
    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (fila < tam && col < tam)
    {
        // Imprimir información de la ejecución de algunos hilos en el bloque
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("[PROCESO CUDA - GPU] Procesando bloque (%d, %d) en el dispositivo.\n", blockIdx.x, blockIdx.y);
        }

        int suma = 0;
        for (int k = 0; k < tam; k++)
        {
            suma += A[fila * tam + k] * B[k * tam + col];
        }
        C[fila * tam + col] = suma;
    }
}

int main()
{
    // Medir tiempo total de ejecución desde el inicio del programa
    struct timeval start_total, end_total;
    gettimeofday(&start_total, NULL);

    // Pre-inicializar el contexto CUDA para evitar retardo durante la fase de multiplicación
    cudaFree(0);

    int i, j;
    int A[N][N];
    int B[N][N];
    int C_GPU[N][N] = {0};

    const char *p_host = "[PROCESO CUDA - HOST]";
    const char *p_gpu = "[PROCESO CUDA - GPU]";

    // 1. Momento en el que el Host inicia
    print_time_info(p_host, "Iniciando programa CUDA y cargando recursos");

    // 2. Generación de números aleatorios por el Host
    srand(time(NULL));
    print_time_info(p_host, "Generación de números aleatorios para matrices A y B");

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    // Mostrar las matrices iniciales
    imprimir_matriz(p_host, A, "A (Inicial)");
    imprimir_matriz(p_host, B, "B (Inicial)");

    // 3. Escribir las matrices en un archivo compartido para que el proceso serial las lea
    print_time_info(p_host, "Guardando matrices A y B en archivo compartido");
    FILE *f = fopen("shared_matrices.tmp", "w");
    if (f == NULL) {
        perror("Error al crear archivo compartido");
        return 1;
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            fprintf(f, "%d ", A[i][j]);
        }
        fprintf(f, "\n");
    }
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            fprintf(f, "%d ", B[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    // Renombrado atómico para evitar race conditions
    rename("shared_matrices.tmp", "shared_matrices.txt");

    // Punteros para el dispositivo (GPU)
    int *devA, *devB, *devC;

    // Reservar memoria en el dispositivo
    cudaMalloc((void**)&devA, N * N * sizeof(int));
    cudaMalloc((void**)&devB, N * N * sizeof(int));
    cudaMalloc((void**)&devC, N * N * sizeof(int));

    // Medición específica de la multiplicación en CUDA (desde la copia hasta el retorno)
    struct timeval start_mul, end_mul;
    gettimeofday(&start_mul, NULL);

    // 4. Copiar matrices al dispositivo
    print_time_info(p_host, "Aviso - Se están enviando las matrices A y B al dispositivo (GPU)");
    
    cudaMemcpy(devA, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devC, C_GPU, N * N * sizeof(int), cudaMemcpyHostToDevice);

    print_time_info(p_gpu, "Aviso - Se recibieron las matrices A y B en el dispositivo (GPU)");

    // 5. Configuración y ejecución de la GPU (Inicia GPU)
    dim3 hilosPorBloque(5, 5);
    dim3 bloquesPorGrid(2, 2);

    print_time_info(p_gpu, "Iniciando ejecución de kernel (Multiplicación en paralelo)");
    
    MultiplicarMatricesGPU<<<bloquesPorGrid, hilosPorBloque>>>(devA, devB, devC, N);
    
    // Esperar a que la GPU termine
    cudaDeviceSynchronize();
    
    print_time_info(p_gpu, "Finalizando ejecución de kernel");

    // 6. Enviar matriz resultado C al Host
    print_time_info(p_gpu, "Aviso - Se envía al host la matriz C");

    cudaMemcpy(C_GPU, devC, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Terminar de medir el tiempo específico de multiplicación
    gettimeofday(&end_mul, NULL);
    double mul_time = (end_mul.tv_sec - start_mul.tv_sec) + 
                      (end_mul.tv_usec - start_mul.tv_usec) / 1000000.0;

    // 7. Host recibe e imprime
    print_time_info(p_host, "Aviso - Se recibió la matriz C de la GPU");
    
    imprimir_matriz(p_host, C_GPU, "C (Resultado GPU)");

    printf("=========================================================\n");
    printf("%s Tiempo específico que tardó la multiplicación en GPU: %.6f segundos\n", p_host, mul_time);
    printf("=========================================================\n\n");

    // Liberar memoria
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    cudaDeviceReset();

    print_time_info(p_host, "Finalizando programa CUDA");

    // Terminar tiempo total de ejecución
    gettimeofday(&end_total, NULL);
    double total_time = (end_total.tv_sec - start_total.tv_sec) + 
                        (end_total.tv_usec - start_total.tv_usec) / 1000000.0;

    printf("=========================================================\n");
    printf("%s Tiempo de ejecución total del programa CUDA (inicio a fin): %.6f segundos\n", p_host, total_time);
    printf("=========================================================\n");

    return 0;
}
