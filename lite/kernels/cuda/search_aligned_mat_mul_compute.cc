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

#include "lite/kernels/cuda/search_aligned_mat_mul_compute.h"

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using SMMFp32 =
    paddle::lite::kernels::cuda::SearchAlignedMatMulCompute<float,
                                                            PRECISION(kFloat)>;
using SMMFp16 =
    paddle::lite::kernels::cuda::SearchAlignedMatMulCompute<half,
                                                            PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(search_aligned_mat_mul, kCUDA, kFloat, kNCHW, SMMFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("_a_addr", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("_b_addr", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("_c_addr", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(search_aligned_mat_mul, kCUDA, kFP16, kNCHW, SMMFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("_a_addr",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("_b_addr",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("_c_addr",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
