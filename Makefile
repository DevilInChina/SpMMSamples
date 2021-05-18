cc=nvcc
NVCC_OPTIONS=-lcublas -lcudart -O3 -gencode=arch=compute_61,code=compute_61

ALL:TIME
	${cc} main.cu time.o -o SpMM ${NVCC_OPTIONS}
TIME:
	gcc time.c -c