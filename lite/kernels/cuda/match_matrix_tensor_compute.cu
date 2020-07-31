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
#include <algorithm>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/match_matrix_tensor_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename dtype>
__global__ void padding_out(const dtype* src,
                            const int* offset,
                            const int seq_num_r,
                            const int max_len_r,
                            const int tl,
                            const int count,
                            dtype* dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int thread_num = blockDim.x * gridDim.x;
  for (tid = threadIdx.x + blockIdx.x * blockDim.x; tid < count;
       tid += thread_num) {
    int seq_id = tid / (tl * max_len_r);
    int tl_id = (tid / (max_len_r)) % tl;
    int r_id = tid % max_len_r;
    int cur_len = offset[seq_id + 1] - offset[seq_id];
    if (r_id < cur_len) {
      dst[tid] = src[(offset[seq_id] + r_id) * tl + tl_id];
    } else {
      dst[tid] = 0.f;
    }
  }
}

template <typename T, PrecisionType PType>
void MatchMatrixTensorCompute<T, PType>::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<T, T>);
}

template <typename T, PrecisionType PType>
void MatchMatrixTensorCompute<T, PType>::Run() {
  CHECK(this->ctx_) << "running context should be set first";
  auto& param = this->template Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();

  auto* x = param.x;
  auto* w = param.w;
  auto* y = param.y;
  auto* out = param.out;
  auto* tmp = param.tmp;
  int dim_t = param.dim_t;
  int dim_in = x->dims()[1];

  const auto& offset_l = x->lod()[0];
  const auto& offset_r = y->lod()[0];
  std::vector<int> offset_r_int(offset_r.size());
  std::transform(offset_r.begin(),
                 offset_r.end(),
                 offset_r_int.begin(),
                 [](int64_t x) -> int { return static_cast<int>(x); });

  int batch = offset_r.size() - 1;
  int len_l = offset_l[1] - offset_l[0];
  for (int i = 1; i < offset_l.size() - 1; i++) {
    int cur_len = offset_l[i + 1] - offset_l[i];
    CHECK_EQ(cur_len, len_l)
        << "each sequence of left matrix is the same length";
  }
  int max_len_r = 0;
  for (int i = 0; i < offset_r.size() - 1; ++i) {
    int cur_len = offset_r[i + 1] - offset_r[i];
    max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
  }

  _input_l_transform.Resize({batch, dim_t, dim_in, len_l});
  _input_l_transform_reorganize.Resize({batch, dim_t, len_l, dim_in});
  _output_tmp.Resize({batch, max_len_r, dim_t, len_l});
  out->Resize({batch, dim_t, len_l, max_len_r});

  _offset_r.Resize({static_cast<int64_t>(offset_r.size())});
  TargetWrapperCuda::MemcpyAsync(_offset_r.mutable_data<int>(TARGET(kCUDA)),
                                 &offset_r_int[0],
                                 sizeof(int) * offset_r.size(),
                                 IoDirection::HtoD,
                                 stream);

  int len_r = offset_r[offset_r.size() - 1];
  const auto* input_l = x->template data<T>();
  const auto* input_r = y->template data<T>();
  const auto* weight_data = w->template data<T>();
  auto* input_l_transform = _input_l_transform.mutable_data<T>(TARGET(kCUDA));
  auto* input_l_transform_reorganize =
      _input_l_transform_reorganize.mutable_data<T>(TARGET(kCUDA));
  auto* output_tmp = _output_tmp.mutable_data<T>(TARGET(kCUDA));
  auto* out_data = out->template mutable_data<T>(TARGET(kCUDA));

  gemm_impl_->init(true, true, dim_t * dim_in, len_l, dim_in, &context);
  gemm_impl_->run(
      1.0f, 0.0f, weight_data, input_l, input_l_transform, &context);
  trans_.transpose(input_l_transform_reorganize,
                   input_l_transform,
                   _input_l_transform.dims().Vectorize(),
                   {0, 1, 3, 2},
                   &stream);
  gemm_impl_->init(false, true, len_r, dim_t * len_l, dim_in, &context);
  gemm_impl_->run(
      1.0f, 0.0f, input_r, input_l_transform_reorganize, output_tmp, &context);
  int seq_num = offset_r.size() - 1;
  int count = seq_num * max_len_r * dim_t * len_l;
  padding_out<T><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
      _output_tmp.data<T>(),
      _offset_r.data<int>(),
      seq_num,
      max_len_r,
      dim_t * len_l,
      count,
      out_data);
  out->set_lod(y->lod());
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using MMTFp32 =
    paddle::lite::kernels::cuda::MatchMatrixTensorCompute<float,
                                                          PRECISION(kFloat)>;
using MMTFp16 =
    paddle::lite::kernels::cuda::MatchMatrixTensorCompute<half,
                                                          PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(match_matrix_tensor, kCUDA, kFloat, kNCHW, MMTFp32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("W",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("Tmp",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(match_matrix_tensor, kCUDA, kFP16, kNCHW, MMTFp16, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindInput("W",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("Tmp",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
