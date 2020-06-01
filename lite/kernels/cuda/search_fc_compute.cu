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
#include "lite/backends/cuda/math/bias.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/search_fc_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType PType>
void SearchFcCompute<T, PType>::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<T, T>);
}

template <typename T, PrecisionType PType>
void SearchFcCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  const Tensor* x_tensor = param.X;
  param.Out->Resize({x_tensor->dims()[0], param.out_size});
  _M = x_tensor->dims().count(0, 1);
  _K = x_tensor->dims().count(1, x_tensor->numel());
  _N = param.out_size;
  const auto* din = x_tensor->data<T>();
  Tensor* out_tensor = param.Out;
  auto* dout = out_tensor->mutable_data<T>(TARGET(kCUDA));
  const auto* weight = param.W->template data<T>();
  const auto* bias = param.b->template data<T>();
  CHECK(gemm_impl_->init(false, true, _M, _N, _K, &ctx));
  gemm_impl_->run(
      __float2half(1.0f), __float2half(0.0f), din, weight, dout, &ctx);

  int total_size = _M * _N;
  lite::cuda::math::add_bias(total_size, _N, bias, dout, stream);
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
