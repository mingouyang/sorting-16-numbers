CUDA_PATH       := /usr/local/cuda
CUDA_INC_PATH   := $(CUDA_PATH)/include
CUDA_BIN_PATH   := $(CUDA_PATH)/bin
CUDA_LIB_PATH   := $(CUDA_PATH)/lib64
NVCC            := $(CUDA_BIN_PATH)/nvcc

INCLUDES  := -I$(CUDA_INC_PATH) -I$(CUDA_PATH)/samples/common/inc
LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart

%: %.cu
	$(NVCC) $(INCLUDES) $(LDFLAGS) -O3 -o $@ $<
