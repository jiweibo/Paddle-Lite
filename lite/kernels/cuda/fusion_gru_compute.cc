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
#include "lite/kernels/cuda/fusion_gru_compute.h"

#include <string>
#include <vector>

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/target_wrapper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType PType>
void FusionGRUCompute<T, PType>::PrepareForRun() {
  auto& context = this->ctx_->template As<CUDAContext>();
  auto& param = this->template Param<param_t>();
  gru_impl_.reset(new lite::cuda::math::CudnnGRU<T, PType>);
  gru_impl_->Init(param, &context);
}

template <typename T, PrecisionType PType>
void FusionGRUCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  gru_impl_->Run(param);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using FusionGRUFp32 =
    paddle::lite::kernels::cuda::FusionGRUCompute<float, PRECISION(kFloat)>;

using FusionGRUFp16 =
    paddle::lite::kernels::cuda::FusionGRUCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(fusion_gru, kCUDA, kFloat, kNCHW, FusionGRUFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Weight_h2h", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Weight_i2h", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Hidden", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(fusion_gru, kCUDA, kFP16, kNCHW, FusionGRUFp16, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("H0", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Weight_h2h",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Weight_i2h",
               {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Hidden",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
