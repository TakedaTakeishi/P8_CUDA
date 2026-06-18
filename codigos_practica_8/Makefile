# Makefile para la Práctica 8: Multiplicación de Matrices (CUDA y Serial)

# Compiladores
NVCC = nvcc
CC = gcc

# Banderas de compilación
CFLAGS = -Wall
NVCCFLAGS =

# Objetivos (Ejecutables)
TARGETS = MultiplicacionMatrices MultiplicacionMatricesCUDA

all: $(TARGETS)

MultiplicacionMatrices: MultiplicacionMatrices.c
	$(CC) $(CFLAGS) MultiplicacionMatrices.c -o MultiplicacionMatrices

MultiplicacionMatricesCUDA: MultiplicacionMatricesCUDA.cu
	$(NVCC) $(NVCCFLAGS) MultiplicacionMatricesCUDA.cu -o MultiplicacionMatricesCUDA

run: all
	@if [ -n "$$DISPLAY" ]; then \
		if command -v gnome-terminal >/dev/null 2>&1; then \
			gnome-terminal --title="PROCESO SERIAL (CPU)" -- bash -c "./MultiplicacionMatrices; echo; read -p 'Presione Enter para salir...'" & \
		elif command -v konsole >/dev/null 2>&1; then \
			konsole --title "PROCESO SERIAL (CPU)" -e bash -c "./MultiplicacionMatrices; echo; read -p 'Presione Enter para salir...'" & \
		else \
			echo "No se encontro gnome-terminal ni konsole. Ejecutando de forma concurrente en la misma ventana..."; \
			./MultiplicacionMatrices & \
		fi; \
		./MultiplicacionMatricesCUDA; \
	else \
		echo "No hay servidor grafico (DISPLAY no definido). Ejecutando de forma concurrente en la misma ventana..."; \
		./MultiplicacionMatricesCUDA & ./MultiplicacionMatrices & wait; \
	fi

clean:
	rm -f $(TARGETS)
