/* The code is described in this paper: Ouyang M, Sorting sixteen
 * numbers.  Proceedings of IEEE High Performance Extreme Computing
 * Conference (HPEC), 2015, 1-6.
 *
 * Copyright (c) 2015 Ming Ouyang
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NoOP      0xFFFFFFFFu
#define TwoTo16   0x00010000u
#define BlockSize 256

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))

int deviceNum = 0;
int dataSize = 16 * TwoTo16;
int *data, *sorted, *gpuData;

void initGPU(void) {
  cudaSetDevice(deviceNum);
  cudaMalloc((void**) &gpuData, sizeof(int) * dataSize);
}

void init(int argc, char *argv[]) {
  unsigned i, j, k, tmp;
  int c;

  while ((c = getopt(argc, argv, "d:")) != -1)
    switch (c) {
    case 'd':
      sscanf(optarg, "%d", &deviceNum);
      break;
    default:
      break;
    }

  data   = (int*) malloc(sizeof(int) * dataSize);
  sorted = (int*) malloc(sizeof(int) * dataSize);

  j = 0;
  for (i = 0; i < TwoTo16; i++) {
    tmp = i;
    for (k = 0; k < 16; k++) {
      data[j++] = tmp & 0x00000001u;
      tmp = tmp >> 1;
    }
  }

  initGPU();
}

void verify(void) {
  unsigned i, j, k, count;

  count = 0;
  for (i = 0; i < TwoTo16; i++) {
    for (j = 0; j < 15; j++) {
      if (sorted[i * 16 + j] > sorted[i * 16 + j + 1]) {
	printf("not sorted %d:", i);

	for (k = 0; k < 16; k++)
	  printf(" %d", sorted[i * 16 + k]);
	printf("\n");

	if (count++ == 10)
	  exit(1);
      }
    }
  }
}

__device__ inline void IntComparator(int &A, int &B) {
  int t;

  if (A > B) {
    t = A;
    A = B;
    B = t;
  }
}

__device__ inline void UnsignedComparator(unsigned &A, unsigned &B) {
  unsigned t;

  if (A > B) {
    t = A;
    A = B;
    B = t;
  }
}

//Nvidia's implementation of Batcher's sorting network
__global__ void nvidiaBatcher(int *X) {
  __shared__ int sX[BlockSize * 2];
  unsigned pos, size, stride, offset;
  unsigned base = blockIdx.x * BlockSize * 2 + threadIdx.x;

  sX[threadIdx.x] = X[base];
  sX[threadIdx.x + BlockSize] = X[base + BlockSize];
  __syncthreads();

#pragma unroll
  for (size = 2; size <= 16; size <<= 1) {
    stride = size >> 1;
    offset = threadIdx.x & (stride - 1);
    pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    IntComparator(sX[pos], sX[pos + stride]);
    stride >>= 1;
    for (; stride > 0; stride >>= 1) {
      pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      if (offset >= stride) //divergent computation
	IntComparator(sX[pos - stride], sX[pos]);
    }
  }
  __syncthreads();

  X[base] = sX[threadIdx.x];
  X[base + BlockSize] = sX[threadIdx.x + BlockSize];
}

//divergent computation in Nvidia's implementation is removed
__global__ void newBatcher(int *X) {
  __shared__ int sX[BlockSize * 2];
  unsigned pos, size, stride, offset;
  unsigned base = blockIdx.x * BlockSize * 2 + threadIdx.x;

  sX[threadIdx.x] = X[base];
  sX[threadIdx.x + BlockSize] = X[base + BlockSize];
  __syncthreads();

#pragma unroll
  for (size = 2; size <= 16; size <<= 1) {
    stride = size >> 1;
    offset = threadIdx.x & (stride - 1);
    pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    IntComparator(sX[pos], sX[pos+stride]);
    stride >>= 1;
    for (; stride > 0; stride >>= 1) {
      pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      //non-divergent computation
      IntComparator(sX[pos - stride * (offset >= stride ? 1 : 0)], sX[pos]);
    }
  }
  __syncthreads();

  X[base] = sX[threadIdx.x];
  X[base + BlockSize] = sX[threadIdx.x + BlockSize];
}

//Van Voorhis's optimal sorting network for 16 numbers
__global__ void VanVoorhis(int *X) {
  __shared__ int sX[BlockSize * 2];
  unsigned wire1, wire2, our16tuple;
  unsigned base = blockIdx.x * BlockSize * 2 + threadIdx.x;

  sX[threadIdx.x] = X[base];
  sX[threadIdx.x + BlockSize] = X[base + BlockSize];
  our16tuple = (threadIdx.x >> 3) << 4;
  __syncthreads();

  //step I
  wire1 = ((threadIdx.x & 7) << 1) + our16tuple;
  wire2 = wire1 + 1;
  IntComparator(sX[wire1], sX[wire2]);

  //step II
  wire1 = (((threadIdx.x & 6) << 1) | (threadIdx.x & 1)) + our16tuple;
  wire2 = wire1 + 2;
  IntComparator(sX[wire1], sX[wire2]);

  //step III
  wire1 = (((threadIdx.x & 4) << 1) | (threadIdx.x & 3)) + our16tuple;
  wire2 = wire1 + 4;
  IntComparator(sX[wire1], sX[wire2]);

  //step IV
  wire1 = (threadIdx.x & 7) + our16tuple;
  wire2 = wire1 + 8;
  IntComparator(sX[wire1], sX[wire2]);

  //step V
  wire1 = threadIdx.x & 7;
  wire1 = (wire1 == 2) ? 13 : wire1;
  wire2 = ((wire1 << 1) & 10) | ((wire1 >> 1) & 5);
  wire2 = (wire1 == wire2) ? (wire2 ^ 15) : wire2;
  wire1 += our16tuple;
  wire2 += our16tuple;
  IntComparator(sX[wire1], sX[wire2]);

  //step VI
  wire1 = threadIdx.x & 7;
  wire2 = ((wire1 << 1) - (wire1 & 1)) << (!(wire1 >> 2));
  wire2 = (wire1 == 0) ? 15 : wire2;
  wire2 = (wire1 == 1) ? 4 : wire2;
  if (wire1 == 4){
    wire1 = wire1 ^ 15;
    wire2 = wire1 + 3;
  }
  wire1 += our16tuple;
  wire2 += our16tuple;
  IntComparator(sX[wire1], sX[wire2]);

  //step VII
  wire1 = (((threadIdx.x & 6) << 1) | (threadIdx.x & 1) | 2) + our16tuple;
  wire2 = (((((threadIdx.x & 6) << 1) | (threadIdx.x & 1) | 2) + 2) & 15)
    + our16tuple;
  UnsignedComparator(wire1, wire2);
  IntComparator(sX[wire1], sX[wire2]);

  //step VIII
  wire1 = ((threadIdx.x & 7) << 1) + 1 + our16tuple;
  wire2 = ((((threadIdx.x & 7) << 1) + 4) & 15) + our16tuple;
  UnsignedComparator(wire1, wire2);
  IntComparator(sX[wire1], sX[wire2]);

  //step IX
  wire1 = ((threadIdx.x & 7) << 1) + 1 + our16tuple;
  wire2 = ((((threadIdx.x & 7) << 1) + 2) & 15) + our16tuple;
  UnsignedComparator(wire1, wire2);
  IntComparator(sX[wire1], sX[wire2]);

  __syncthreads();
  X[base] = sX[threadIdx.x];
  X[base + BlockSize] = sX[threadIdx.x + BlockSize];
}

int main(int argc, char *argv[]) {
  cudaEvent_t start;
  cudaEvent_t stop;
  float msec;

  init(argc, argv);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(gpuData, data, sizeof(int) * dataSize, cudaMemcpyHostToDevice);
  cudaEventRecord(start, NULL);
  nvidiaBatcher <<<dataSize / (BlockSize * 2), BlockSize>>> (gpuData);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msec, start, stop);
  printf("Nvd Batcher, blocksize %d, device %d, %.3f ms\n",
	 BlockSize, deviceNum, msec);
  cudaMemcpy(sorted, gpuData, sizeof(int) * dataSize, cudaMemcpyDeviceToHost);
  verify();

  cudaMemcpy(gpuData, data, sizeof(int) * dataSize, cudaMemcpyHostToDevice);
  cudaEventRecord(start, NULL);
  VanVoorhis <<<dataSize / (BlockSize * 2), BlockSize>>> (gpuData);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msec, start, stop);
  printf("Van Voorhis, blocksize %d, device %d, %.3f ms\n",
	 BlockSize, deviceNum, msec);
  cudaMemcpy(sorted, gpuData, sizeof(int) * dataSize, cudaMemcpyDeviceToHost);
  verify();

  cudaMemcpy(gpuData, data, sizeof(int) * dataSize, cudaMemcpyHostToDevice);
  cudaEventRecord(start, NULL);
  newBatcher <<<dataSize / (BlockSize * 2), BlockSize>>> (gpuData);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msec, start, stop);
  printf("new Batcher, blocksize %d, device %d, %.3f ms\n",
	 BlockSize, deviceNum, msec);
  cudaMemcpy(sorted, gpuData, sizeof(int) * dataSize, cudaMemcpyDeviceToHost);
  verify();

  return 0;
}
