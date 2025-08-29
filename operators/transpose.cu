#include <cuda_runtime.h>
#include <iostream>

__global__ void naiveGmem(float *out, float *in, int nx, int ny) {
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
  if (ix < nx && iy < ny) {
    out[ix * ny + iy] = in[iy * nx + ix];
  }
}

void call_naiveGmem(float *d_out, float *d_in, int nx, int ny) {
  dim3 blockSize(2, 2); 
  dim3 gridSize((nx + blockSize.x - 1) / blockSize.x,
                (ny + blockSize.y - 1) / blockSize.y);
  naiveGmem<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
}

int main() {
  int nx = 4;
  int ny = 4;
  size_t size = nx * ny * sizeof(float);


  float *h_in = (float *)malloc(size);
  float *h_out = (float *)malloc(size);

  for (int i = 0; i < nx * ny; i++) {
    h_in[i] = float(int(i) % 11);
  }

  float *d_in, *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);


  cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

  call_naiveGmem(d_out, d_in, nx, ny);

  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      std::cout << h_in[j * nx + i] << " ";
    }
    std::cout << "\n";
  }

  printf("---------------\n");

  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      std::cout << h_out[j * nx + i] << " ";
    }
    std::cout << "\n";
  }

  free(h_in);
  free(h_out);
  cudaFree(d_in);
  cudaFree(d_out);

  std::cout << "Matrix transposition completed." << std::endl;

  return 0;
}