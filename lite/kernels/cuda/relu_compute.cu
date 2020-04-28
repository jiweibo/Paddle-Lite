// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/cuda/math/activation.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/relu_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

// template <typename T>
// __global__ void ReluKernel(const int num, const T* input, T* output) {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < num) {
// #if __CUDA_ARCH__ >= 350
//     output[index] = __ldg(input + index) >= 0 ? __ldg(input + index) : 0;
// #else
//     output[index] = input[index] >= 0 ? input[index] : 0;
// #endif
//   }
// }

template <typename T, PrecisionType PType>
void ReluCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  int num = static_cast<int>(param.X->numel());
  auto input = param.X->template data<T>();
  auto output = param.Out->template mutable_data<T>(TARGET(kCUDA));

  lite::cuda::math::relu(num, input, output, 0.f, stream);

  // int threads = 1024;
  // int blocks = (num + threads - 1) / threads;
  // ReluKernel<<<blocks, threads, 0, stream>>>(num, input, output);
  // cudaError_t error = cudaGetLastError();
  // if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using ReLUFp32 =
    paddle::lite::kernels::cuda::ReluCompute<float, PRECISION(kFloat)>;
using ReLUFp16 =
    paddle::lite::kernels::cuda::ReluCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(relu, kCUDA, kFloat, kNCHW, ReLUFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(relu, kCUDA, kFP16, kNCHW, ReLUFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
