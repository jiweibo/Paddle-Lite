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

#include <vector>
#include "lite/backends/cuda/math/type_trans.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/lookup_table_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {
using Tensor = lite::Tensor;

template <typename T>
__global__ void LookupTableKernel(T *output,
                                  const T *table,
                                  const int64_t *ids,
                                  const int64_t N,
                                  const int64_t K,
                                  const int64_t D,
                                  const bool padding_flag,
                                  const int64_t padding_idx);

template <>
__global__ void LookupTableKernel<float>(float *output,
                                         const float *table,
                                         const int64_t *ids,
                                         const int64_t N,
                                         const int64_t K,
                                         const int64_t D,
                                         const bool padding_flag,
                                         const int64_t padding_idx) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * gridDim.x;
  while (idy < K) {
    int64_t id = ids[idy];
    float *out = output + idy * D;
    const float *tab = table + id * D;
    for (int i = idx; i < D; i += blockDim.x) {
      if (padding_flag) {
        if (id == padding_idx)
          out[i] = 0.f;
        else
          out[i] = tab[i];
      } else {
        out[i] = tab[i];
      }
    }
    idy += blockDim.y * gridDim.x;
  }
}

template <>
__global__ void LookupTableKernel<half>(half *output,
                                        const half *table,
                                        const int64_t *ids,
                                        const int64_t N,
                                        const int64_t K,
                                        const int64_t D,
                                        const bool padding_flag,
                                        const int64_t padding_idx) {
  int idx = threadIdx.x;
  int idy = blockIdx.x + threadIdx.y * gridDim.x;
  while (idy < K) {
    int64_t id = ids[idy];
    half *out = output + idy * D;
    const half *tab = table + id * D;
#if __CUDA_ARCH__ >= 530
    // if D is not an integer multiple of 2, cuda will triger the problem of
    // misaligned address problem.
    if (D % 2) {
      for (int i = idx; i < D; i += blockDim.x) {
        if (padding_flag) {
          if (id == padding_idx)
            out[i] = __float2half(0.f);
          else
            out[i] = tab[i];
        } else {
          out[i] = tab[i];
        }
      }
    } else {
      int D2 = D / 2;
      half2 *out2 = reinterpret_cast<half2 *>(out);
      const half2 *table2 = reinterpret_cast<const half2 *>(tab);
      for (int i = idx; i < D2; i += blockDim.x) {
        if (padding_flag) {
          if (id == padding_idx) {
            out2[i] = __float2half2_rn(0.f);
          } else {
            out2[i] = table2[i];
          }
        }
      }
      if (idx == 0 && D % 2) {
        out[D - 1] = tab[D - 1];
      }
    }
#else
    for (int i = idx; i < D; i += blockDim.x) {
      if (padding_flag) {
        if (id == padding_idx)
          out[i] = __float2half(0.f);
        else
          out[i] = tab[i];
      } else {
        out[i] = tab[i];
      }
    }
#endif
    idy += blockDim.y * gridDim.x;
  }
}

template <>
void LookupTableCompute<float, PRECISION(kFloat)>::PrepareForRun() {
  auto &param = this->Param<param_t>();
  w_tensor_ = param.W;
}

template <>
void LookupTableCompute<half, PRECISION(kFP16)>::PrepareForRun() {
  auto &param = this->Param<param_t>();
  w_half_tensor_.Resize(param.W->dims());
  lite::cuda::math::fp32_to_fp16(
      param.W->numel(),
      param.W->data<float>(),
      w_half_tensor_.mutable_data<half>(TARGET(kCUDA)));
  w_half_tensor_.set_lod(param.W->lod());
  w_tensor_ = &w_half_tensor_;
}

template <typename T, PrecisionType PType>
void LookupTableCompute<T, PType>::Run() {
  auto &param = this->template Param<param_t>();
  auto &ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();
  const Tensor *ids_t = param.Ids;
  Tensor *out_t = param.Out;
  int64_t padding_idx = param.padding_idx;

  size_t N = w_tensor_->dims()[0];
  size_t D = w_tensor_->dims()[1];
  size_t K = ids_t->numel();

  auto *w = w_tensor_->data<T>();
  auto *ids = ids_t->data<int64_t>();
  auto *out = out_t->mutable_data<T>(TARGET(kCUDA));

  dim3 threads(128, 8);
  dim3 grids(8, 1);
  bool padding_flag = !(padding_idx == -1);
  LookupTableKernel<T><<<grids, threads, 0, stream>>>(
      out, w, ids, N, K, D, padding_flag, padding_idx);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using LKTFp32 =
    paddle::lite::kernels::cuda::LookupTableCompute<float, PRECISION(kFloat)>;
using LKTFp16 =
    paddle::lite::kernels::cuda::LookupTableCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(lookup_table, kCUDA, kFloat, kNCHW, LKTFp32, def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .Finalize();
REGISTER_LITE_KERNEL(lookup_table_v2, kCUDA, kFloat, kNCHW, LKTFp32, def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .Finalize();
REGISTER_LITE_KERNEL(lookup_table, kCUDA, kFP16, kNCHW, LKTFp16, def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
REGISTER_LITE_KERNEL(lookup_table_v2, kCUDA, kFP16, kNCHW, LKTFp16, def)
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFloat))})
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
