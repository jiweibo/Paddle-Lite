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

#include <vector>

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/kernels/cuda/attention_padding_mask_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T>
__global__ void ker_attention_padding_mask(T* out_data,
                                           const T* attn_data,
                                           const int* src_offset,
                                           const int attn_seq_num,
                                           const int attn_seq_len,
                                           const int src_seq_num,
                                           const int src_seq_len,
                                           const int* pad_begin_data,
                                           const T mask,
                                           const int count) {
  CUDA_KERNEL_LOOP(tid, count) {
    int src_word_id = tid % src_seq_len;
    int tmp_tid = tid / src_seq_len;
    int attn_seq_id = tmp_tid / attn_seq_len;
    int attn_word_id = tmp_tid % attn_seq_len;
    int src_seq_id = attn_seq_id % src_seq_num;
    int cur_len = src_offset[src_seq_id + 1] - src_offset[src_seq_id];

    int k = pad_begin_data[src_seq_id];
    if (k < cur_len &&
        tid >= src_seq_len * (attn_seq_len * attn_seq_id + attn_word_id) + k &&
        tid < src_seq_len * (attn_seq_len * attn_seq_id + attn_word_id) +
                  cur_len) {
      out_data[tid] = mask;
    } else {
      out_data[tid] = attn_data[tid];
    }
  }
}

template <typename Dtype>
__global__ void ker_find_begin_data(int count,
                                    int* out,
                                    const Dtype* src,
                                    const Dtype pad_data,
                                    const int offset_len) {
  CUDA_KERNEL_LOOP(tid, count) {
    int index = offset_len - 1;
    const Dtype* src_data = src + offset_len * tid;
    for (; index >= 0 && pad_data == src_data[index]; --index) {
    }
    out[tid] = index + 1;
  }
}

template <typename Dtype, PrecisionType Ptype>
void AttentionPaddingMaskCompute<Dtype, Ptype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  auto attn = param.X;
  auto src = param.Y;
  const int count = attn->numel();
  auto attn_offset = attn->lod()[0];
  auto src_offset = src->lod()[0];
  const int attn_seq_num = attn_offset.size() - 1;
  const int attn_seq_len = attn_offset[1];
  const int src_seq_num = src_offset.size() - 1;
  const int src_seq_len = count / attn->dims()[0];

  auto out = param.Out;
  out->Resize(attn->dims());
  out->set_lod(attn->lod());

  auto attn_data = attn->template data<Dtype>();
  auto out_data = out->template mutable_data<Dtype>(TARGET(kCUDA));

  param.pad_begin->Resize({static_cast<int64_t>(src_seq_num)});
  auto pad_begin_cuda_data =
      param.pad_begin->template mutable_data<int>(TARGET(kCUDA));
  ker_find_begin_data<
      Dtype><<<CUDA_GET_BLOCKS(src_seq_num), CUDA_NUM_THREADS, 0, stream>>>(
      src_seq_num,
      pad_begin_cuda_data,
      src->template data<Dtype>(),
      static_cast<float>(param.pad_id),
      static_cast<int>(src->lod()[0][1]));

  std::vector<int> src_offset_cpu(src_offset.size(), 0);
  for (int i = 0; i < src_offset.size(); i++) {
    src_offset_cpu[i] = src_offset[i];
  }

  src_offset_cuda.Resize({static_cast<int64_t>(src_offset.size())});
  auto src_offset_cuda_data = src_offset_cuda.mutable_data<int>(TARGET(kCUDA));
  TargetWrapperCuda::MemcpyAsync(src_offset_cuda_data,
                                 src_offset_cpu.data(),
                                 sizeof(int) * src_offset.size(),
                                 IoDirection::HtoD,
                                 stream);

  ker_attention_padding_mask<
      Dtype><<<CUDA_GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
      out_data,
      attn_data,
      src_offset_cuda_data,
      attn_seq_num,
      attn_seq_len,
      src_seq_num,
      src_seq_len,
      pad_begin_cuda_data,
      param.mask,
      count);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(ERROR) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using APMFp32 =
    paddle::lite::kernels::cuda::AttentionPaddingMaskCompute<float,
                                                             PRECISION(kFloat)>;
using APMFp16 =
    paddle::lite::kernels::cuda::AttentionPaddingMaskCompute<half,
                                                             PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(
    search_attention_padding_mask, kCUDA, kFloat, kNCHW, APMFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("pad_begin",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .Finalize();

REGISTER_LITE_KERNEL(
    search_attention_padding_mask, kCUDA, kFP16, kNCHW, APMFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("pad_begin",
                {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kInt64))})
    .Finalize();
