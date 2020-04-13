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
#include "lite/backends/cuda/math/type_trans.h"
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
__global__ void add_bias<half>(int n,
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

template <>
void SearchFcCompute<float, PRECISION(kFloat)>::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<float, float>);
  auto& param = this->Param<param_t>();
  w_tensor_ = param.W;
  b_tensor_ = param.b;
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
  const float* weight = w_tensor_->data<float>();
  const float* bias = b_tensor_->data<float>();
  CHECK(gemm_impl_->init(false, true, _M, _N, _K, &ctx));
  gemm_impl_->run(1.0f, 0.0f, din, weight, dout, &ctx);

  int total_size = _M * _N;
  add_bias<float><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>(
      total_size, _N, bias, dout);
}

template <>
void SearchFcCompute<half, PRECISION(kFP16)>::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<half, half>);
  auto& param = this->Param<param_t>();
  w_tensor_ = param.W;
  b_tensor_ = param.b;
  w_half_tensor_.Resize(w_tensor_->dims());
  lite::cuda::math::fp32_to_fp16(
      w_tensor_->numel(),
      w_tensor_->data<float>(),
      w_half_tensor_.mutable_data<half>(TARGET(kCUDA)));
  w_half_tensor_.set_lod(w_tensor_->lod());
  b_half_tensor_.Resize(b_tensor_->dims());
  lite::cuda::math::fp32_to_fp16(
      b_tensor_->numel(),
      b_tensor_->data<float>(),
      b_half_tensor_.mutable_data<half>(TARGET(kCUDA)));
  b_half_tensor_.set_lod(b_tensor_->lod());
}

template <>
void SearchFcCompute<half, PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  const Tensor* x_tensor = param.X;
  param.Out->Resize({x_tensor->dims()[0], param.out_size});
  _M = x_tensor->dims().count(0, 1);
  _K = x_tensor->dims().count(1, x_tensor->numel());
  _N = param.out_size;
  const half* din = x_tensor->data<half>();
  Tensor* out_tensor = param.Out;
  half* dout = out_tensor->mutable_data<half>(TARGET(kCUDA));
  const __half* weight = w_half_tensor_.data<__half>();
  const __half* bias = b_half_tensor_.data<__half>();
  CHECK(gemm_impl_->init(false, true, _M, _N, _K, &ctx));
  gemm_impl_->run(
      __float2half(1.0f), __float2half(0.0f), din, weight, dout, &ctx);

  int total_size = _M * _N;
  add_bias<half><<<CUDA_GET_BLOCKS(total_size), CUDA_NUM_THREADS, 0, stream>>>(
      total_size, _N, bias, dout);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using FCFp32 =
    paddle::lite::kernels::cuda::SearchFcCompute<float, PRECISION(kFloat)>;
using FCFp16 =
    paddle::lite::kernels::cuda::SearchFcCompute<half, PRECISION(kFP16)>;

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
