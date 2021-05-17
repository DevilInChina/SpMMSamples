cc=nvcc
NVCC_OPTIONS=-lcublas -lcudart -O3 -gencode=arch=compute_61,code=compute_61

ALL:
	${cc} main.cu -o SpMM ${NVCC_OPTIONS}