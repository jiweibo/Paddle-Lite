/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/search_fc_compute.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
__global__ void add_bias(int n, int output_size, const T* bias, T* dout) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int bias_index = index % output_size;
  if (index < n) {
    dout[index] = dout[index] + bias[bias_index];
  }
}

template <>
__global__ void add_bias<__half>(int n,
                                 int output_size,
                                 const __half* bias,
                                 __half* dout) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
#if __CUDA_ARCH__ >= 530
  int n2 = n / 2;
  if (index < n2) {
    __half2* dout2 = reinterpret_cast<__half2*>(dout);
    __half2 bias_data;
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

template <typename T, PrecisionType PType>
void SearchFcCompute<T, PType>::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<T, T>);
}

template <>
void SearchFcCompute<float, PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  const Tensor* x_tensor = param.X;
  param.Out->Resize({x_tensor->dims()[0], param.out_size});
  _M = x_tensor->dims().count(0, 1);
  _K = x_tensor->dims().count(1, x_tensor->numel());
  _N = param.out_size;
  const float* din = x_tensor->data<float>();
  Tensor* out_tensor = param.Out;
  float* dout = out_tensor->mutable_data<float>(TARGET(kCUDA));
  const Tensor* w_tensor = param.W;
  const float* weight = w_tensor->data<float>();
  const Tensor* b_tensor = param.b;
  const float* bias = b_tensor->data<float>();
  CHECK(gemm_impl_->init(false, true, _M, _N, _K, &ctx));
  gemm_impl_->run(1.0f, 0.0f, din, weight, dout, &ctx);

  int total_size = _M * _N;
  add_bias<float><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>(
      total_size, _N, bias, dout);
}

template <>
void SearchFcCompute<__half, PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  const Tensor* x_tensor = param.X;
  param.Out->Resize({x_tensor->dims()[0], param.out_size});
  _M = x_tensor->dims().count(0, 1);
  _K = x_tensor->dims().count(1, x_tensor->numel());
  _N = param.out_size;
  const __half* din = x_tensor->data<__half>();
  Tensor* out_tensor = param.Out;
  __half* dout = out_tensor->mutable_data<__half>(TARGET(kCUDA));
  const Tensor* w_tensor = param.W;
  const __half* weight = w_tensor->data<__half>();
  const Tensor* b_tensor = param.b;
  const __half* bias = b_tensor->data<__half>();
  CHECK(gemm_impl_->init(false, true, _M, _N, _K, &ctx));
  gemm_impl_->run(
      __float2half(1.0f), __float2half(0.0f), din, weight, dout, &ctx);

  int total_size = _M * _N;
  add_bias<
      __half><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>(
      total_size, _N, bias, dout);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using FCFp32 =
    paddle::lite::kernels::cuda::SearchFcCompute<float, PRECISION(kFloat)>;
using FCFp16 =
    paddle::lite::kernels::cuda::SearchFcCompute<__half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(search_fc, kCUDA, kFloat, kNCHW, FCFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(search_fc, kCUDA, kFP16, kNCHW, FCFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
