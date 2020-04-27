// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iostream"
#include "lite/backends/cuda/math/bias.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename dtype>
__global__ void bias_kernel(int n,
                            int output_size,
                            const dtype* bias,
                            dtype* dout) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias_index = index % output_size;
  if (index < n) {
    dout[index] = dout[index] + bias[bias_index];
  }
}

template <>
__global__ void bias_kernel<half>(int n,
                                  int output_size,
                                  const half* bias,
                                  half* dout) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
#if __CUDA_ARCH__ >= 530
  int n2 = n / 2;
  if (index < n2) {
    half2* dout2 = reinterpret_cast<half2*>(dout);
    half2 bias_data;
    bias_data.x = bias[(2 * index) % output_size];
    bias_data.y = bias[(2 * index + 1) % output_size];
    dout2[index] = __hadd2(dout2[index], bias_data);
  }
  if (index == 0 && n % 2) {
    dout[n - 1] = __hadd(dout[n - 1], bias[(n - 1) % output_size]);
  }
#else
  if (index < n) {
    dout[index] = __float2half(__half2float(dout[index]) +
                               __half2float(bias[index % output_size]));
  }
#endif
}

template <typename T>
void add_bias(int num,
              int output_size,
              const T* bias_data,
              T* out_data,
              cudaStream_t stream) {
  bias_kernel<T><<<CUDA_GET_BLOCKS(num), CUDA_NUM_THREADS, 0, stream>>>(
      num, output_size, bias_data, out_data);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) std::cout << cudaGetErrorString(error);
}

template void add_bias(int num,
                       int output_size,
                       const float* bias_data,
                       float* out_data,
                       cudaStream_t stream);
template void add_bias(int num,
                       int output_size,
                       const half* bias_data,
                       half* out_data,
                       cudaStream_t stream);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
