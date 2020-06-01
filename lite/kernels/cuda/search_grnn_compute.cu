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
#include "lite/backends/cuda/math/transpose.h"
#include "lite/backends/cuda/math/type_trans.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/search_grnn_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using Tensor = lite::Tensor;

template <typename Dtype>
__global__ void trans_map2out(
    Dtype* output, const Dtype* input, const int* map, int count, int lastdim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    int seq = tid / lastdim;
    output[map[seq] * lastdim + tid % lastdim] = input[tid];
  }
}

template <typename Dtype>
__global__ void trans_map2in(
    Dtype* output, const Dtype* input, const int* map, int count, int lastdim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < count) {
    int seq = tid / lastdim;
    output[tid] = input[map[seq] * lastdim + tid % lastdim];
  }
}

template <typename Dtype>
void trans_map2out_cfunc(const Dtype* input,
                         Dtype* output,
                         int word_size,
                         int seq_sum,
                         cudaStream_t stream,
                         int* dev_map_vec) {
  int count = seq_sum * word_size;
  int block_dim = count;
  int grid_dim = 1;

  if (count > 1024) {
    block_dim = 256;
    grid_dim = (count + block_dim - 1) / block_dim;
  }

  trans_map2out<<<grid_dim, block_dim, 0, stream>>>(
      output, input, dev_map_vec, count, word_size);
}

template <typename Dtype>
void trans_map2in_cfunc(const Dtype* input,
                        Dtype* output,
                        int hidden_size,
                        int seq_sum,
                        cudaStream_t stream,
                        int* dev_map_vec) {
  int count = seq_sum * hidden_size;
  int block_dim = count;
  int grid_dim = 1;
  if (count > 1024) {
    block_dim = 256;
    grid_dim = (count + block_dim - 1) / block_dim;
  }

  trans_map2in<<<grid_dim, block_dim, 0, stream>>>(
      output, input, dev_map_vec, count, hidden_size);
}

template <typename Dtype>
void SeqSortedseqTranseUtil::seq_2_sorted_seq(const Dtype* input,
                                              Dtype* output,
                                              int word_size,
                                              cudaStream_t stream) {
  int seq_sum = _map_vec.size();
  trans_map2out_cfunc(input, output, word_size, seq_sum, stream, _dev_map_vec);
}

template <typename Dtype>
void SeqSortedseqTranseUtil::sorted_seq_2_seq(const Dtype* input,
                                              Dtype* output,
                                              int hidden_size,
                                              cudaStream_t stream) {
  int seq_sum = _map_vec.size();
  trans_map2in_cfunc(input, output, hidden_size, seq_sum, stream, _dev_map_vec);
}

bool SeqSortedseqTranseUtil::get_sorted_map(const std::vector<int>& offset_vec,
                                            cudaStream_t stream_id) {
  int batch_size = offset_vec.size() - 1;
  int word_sum = offset_vec[offset_vec.size() - 1];
  std::vector<int> length_vec(batch_size);
  _length_index.resize(batch_size);
  int emit_length = 0;

  if (batch_size == 1) {
    emit_length = offset_vec[1] - offset_vec[0];
    _emit_offset_vec.resize(emit_length + 1);

    for (int i = 0; i <= emit_length; ++i) {
      _emit_offset_vec[i] = i;
    }

    return false;
  }

  int max_len = 0;

  for (int i = 0; i < offset_vec.size() - 1; ++i) {
    int len = offset_vec[i + 1] - offset_vec[i];
    max_len = max_len > len ? max_len : len;
    length_vec[i] = len;
    _length_index[i] = i;
  }

  emit_length = max_len;

  if (max_len == 1) {
    _emit_offset_vec.resize(2);
    _emit_offset_vec[0] = 0;
    _emit_offset_vec[1] = emit_length * batch_size;
    return false;
  }

  std::stable_sort(_length_index.begin(),
                   _length_index.end(),
                   [&length_vec](int i1, int i2) {
                     return length_vec[i1] > length_vec[i2];
                   });

  _emit_offset_vec.resize(max_len + 1);
  _map_vec.resize(word_sum);

  if (word_sum > _dev_map_vec_length) {
    if (_dev_map_vec != nullptr) {
      TargetWrapperCuda::Free(static_cast<void*>(_dev_map_vec));
    }

    _dev_map_vec =
        static_cast<int*>(TargetWrapperCuda::Malloc(sizeof(int) * word_sum));
    _dev_map_vec_length = word_sum;
  }

  int target_word_id = 0;
  std::vector<int> length_vec_cnt = length_vec;
  int last_batch_size = batch_size;
  for (int word_id_in_seq = 0; word_id_in_seq < max_len; word_id_in_seq++) {
    _emit_offset_vec[word_id_in_seq] = target_word_id;

    for (int batch_id = 0; batch_id < last_batch_size; batch_id++) {
      int old_batch_id = _length_index[batch_id];

      if (length_vec_cnt[old_batch_id] > 0) {
        int inner_word_id_in_seq = word_id_in_seq;

        if (_is_reverse) {
          inner_word_id_in_seq = length_vec[old_batch_id] - 1 - word_id_in_seq;
        }

        int old_word_id = offset_vec[old_batch_id] + inner_word_id_in_seq;
        _map_vec[old_word_id] = target_word_id;
        length_vec_cnt[old_batch_id]--;
        target_word_id++;
      } else {
        last_batch_size--;
        break;
      }
    }
  }

  TargetWrapperCuda::MemcpyAsync(_dev_map_vec,
                                 _map_vec.data(),
                                 sizeof(int) * word_sum,
                                 IoDirection::HtoD,
                                 stream_id);
  _emit_offset_vec[max_len] = word_sum;
  _emit_length = emit_length;
  return true;
}

template <typename Dtype>
__global__ void transpose_2d(Dtype* output, const Dtype* input, int m, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < m * n) {
    int i = tid / n;
    int j = tid % m;
    output[tid] = input[j * n + i];
  }
}

template <>
void SearchGrnnCompute<float, PRECISION(kFloat)>::WeightsPreprocess() {
  auto& param = this->Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();

  DDim idims = param.wi->dims();
  DDim hdims = param.wh->dims();
  _wi.Resize({idims[2], idims[0], idims[1]});
  _wh.Resize({hdims[2], hdims[0], hdims[1]});
  lite::cuda::math::Transpose<float> trans;
  trans.transpose(_wi.mutable_data<float>(TARGET(kCUDA)),
                  param.wi->data<float>(),
                  idims.Vectorize(),
                  {2, 0, 1},
                  &stream);
  trans.transpose(_wh.mutable_data<float>(TARGET(kCUDA)) + hdims[1] * hdims[2],
                  param.wh->data<float>() + hdims[1] * hdims[2],
                  {hdims[0] - 1, hdims[1], hdims[2]},
                  {2, 0, 1},
                  &stream);
  trans.transpose(_wh.mutable_data<float>(TARGET(kCUDA)),
                  param.wh->data<float>(),
                  {hdims[1], hdims[2]},
                  {1, 0},
                  &stream);

  // int thread_num = 512;
  // int block_num = (hdims[1] * hdims[2] + thread_num - 1) / thread_num;
  // transpose_2d<<<block_num, thread_num, 0, stream>>>(
  //    _wh.mutable_data<float>(TARGET(kCUDA)),
  //    param.wh->data<float>(),
  //    hdims[1],
  //    hdims[2]);
}

template <>
void SearchGrnnCompute<half, PRECISION(kFP16)>::WeightsPreprocess() {
  auto& param = this->Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();

  DDim idims = param.wi->dims();
  DDim hdims = param.wh->dims();
  _wi.Resize({idims[2], idims[0], idims[1]});
  _wi_tmp.Resize({idims[2], idims[0], idims[1]});
  _wh.Resize({hdims[2], hdims[0], hdims[1]});
  _wh_tmp.Resize({hdims[2], hdims[0], hdims[1]});
  lite::cuda::math::Transpose<float> trans;
  trans.transpose(_wi_tmp.mutable_data<float>(TARGET(kCUDA)),
                  param.wi->data<float>(),
                  idims.Vectorize(),
                  {2, 0, 1},
                  &stream);
  trans.transpose(
      _wh_tmp.mutable_data<float>(TARGET(kCUDA)) + hdims[1] * hdims[2],
      param.wh->data<float>() + hdims[1] * hdims[2],
      {hdims[0] - 1, hdims[1], hdims[2]},
      {2, 0, 1},
      &stream);
  trans.transpose(_wh_tmp.mutable_data<float>(TARGET(kCUDA)),
                  param.wh->data<float>(),
                  {hdims[1], hdims[2]},
                  {1, 0},
                  &stream);

  lite::cuda::math::fp32_to_fp16(_wh_tmp.numel(),
                                 _wh_tmp.data<float>(),
                                 _wh.mutable_data<half>(TARGET(kCUDA)),
                                 stream);
  lite::cuda::math::fp32_to_fp16(_wi_tmp.numel(),
                                 _wi_tmp.data<float>(),
                                 _wi.mutable_data<half>(TARGET(kCUDA)),
                                 stream);
}

template <typename T, PrecisionType PType>
void SearchGrnnCompute<T, PType>::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();
  gemm_impl_.reset(new lite::cuda::math::Gemm<T, T>);
  _seq_util = SeqSortedseqTranseUtil();

  WeightsPreprocess();

  int hidden_size = param.num_hidden;
  int word_size = param.num_input;
  int weights_h2h_size = hidden_size * hidden_size * 3;

  lite::Tensor temp_weights_h2h_ori;
  lite::Tensor temp_weights_h2h_swarp;
  temp_weights_h2h_ori.Resize({weights_h2h_size});
  temp_weights_h2h_swarp.Resize({weights_h2h_size});

  TargetWrapperCuda::MemcpyAsync(temp_weights_h2h_ori.mutable_data<T>(),
                                 _wh.data<T>(),
                                 sizeof(T) * weights_h2h_size,
                                 IoDirection::DtoH,
                                 stream);
  cudaStreamSynchronize(stream);

  T* temp_tensor_ptr = temp_weights_h2h_swarp.mutable_data<T>();
  memcpy(temp_tensor_ptr,
         temp_weights_h2h_ori.data<T>(),
         sizeof(T) * hidden_size * hidden_size);

  T* rz_temp_tensor_ptr = temp_tensor_ptr + hidden_size * hidden_size;
  const T* rz_weights_tensor_ptr =
      temp_weights_h2h_ori.data<T>() + hidden_size * hidden_size;
  for (int row = 0; row < hidden_size; row++) {
    for (int block = 0; block < 2; block++) {
      int block_offset = block * hidden_size;
      for (int cow = 0; cow < hidden_size; cow++) {
        rz_temp_tensor_ptr[block * hidden_size * hidden_size +
                           row * hidden_size + cow] =
            rz_weights_tensor_ptr[row * (2 * hidden_size) + cow + block_offset];
      }
    }
  }

  T* orz_temp_tensor_ptr = temp_tensor_ptr;
  T* orz_weights_tensor_ptr = temp_weights_h2h_ori.mutable_data<T>();
  for (int row = 0; row < hidden_size; row++) {
    for (int block = 0; block < 3; block++) {
      int block_offset = block * hidden_size;
      for (int cow = 0; cow < hidden_size; cow++) {
        orz_weights_tensor_ptr[row * (3 * hidden_size) + cow + block_offset] =
            orz_temp_tensor_ptr[block * hidden_size * hidden_size +
                                row * hidden_size + cow];
      }
    }
  }

  _temp_weights_h2h.Resize({weights_h2h_size});
  TargetWrapperCuda::MemcpyAsync(
      _temp_weights_h2h.mutable_data<T>(TARGET(kCUDA)),
      temp_weights_h2h_ori.data<T>(),
      sizeof(T) * weights_h2h_size,
      IoDirection::HtoD,
      stream);
  cudaStreamSynchronize(stream);
}

template <typename Dtype>
static inline __device__ Dtype Sigmoid(const Dtype a) {
  return static_cast<Dtype>(1.0) / (static_cast<Dtype>(1.0) + expf(-a));
}

template <>
__device__ half Sigmoid<half>(const half a) {
#ifdef __CUDA_ARCH__ >= 530
  const half tmp = __float2half(1.0f);
  return __hdiv(tmp, __hadd(tmp, hexp(__hmul(__float2half(-1.f), a))));
#else
  return __float2half(1.0f / (expf(__half2float(a) * -1) + 1.0f));
#endif
}

template <typename Dtype>
static inline __device__ Dtype Tanh(const Dtype a) {
  Dtype tmp = static_cast<Dtype>(-2.0) * a;
  return (static_cast<Dtype>(2.0) / (static_cast<Dtype>(1.0) + expf(tmp))) -
         static_cast<Dtype>(1.0);
}

template <>
__device__ half Tanh(const half a) {
#ifdef __CUDA_ARCH__ >= 530
  half tmp = __float2half(1.0f);
  half numerator = __hmul(__float2half(-2.0f), a);
  return __hsub(__hdiv(__float2half(2.0f), __hadd(tmp, hexp(numerator))), tmp);
#else
  float tmp = -2.0f * __half2float(a);
  return __float2half(2.0f / (1.0f + expf(tmp)) - 1.0f);
#endif
}

template <typename Dtype>
__global__ void cal_cudnn_kernel(const Dtype* w_x_r,
                                 const Dtype* w_x_z,
                                 const Dtype* w_x_o,
                                 const Dtype* w_h_r,
                                 const Dtype* w_h_z,
                                 const Dtype* w_h_o,
                                 int hidden_size,
                                 int batch_size,
                                 Dtype* output,
                                 const Dtype* hidden_pre) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_id = thread_id / hidden_size;
  const int index = thread_id % hidden_size;
  if (index < hidden_size && batch_id < batch_size) {
    int w_base_index = batch_id * hidden_size * 3 + index;
    int h_base_index = batch_id * hidden_size + index;
    Dtype hidden_pre_value = hidden_pre[h_base_index];
    Dtype r = Sigmoid(w_x_r[w_base_index] + w_h_r[w_base_index]);
    Dtype z = Sigmoid(w_x_z[w_base_index] + w_h_z[w_base_index]);
    Dtype _h = Tanh(w_x_o[w_base_index] + w_h_o[w_base_index] * r);

    output[h_base_index] = (1.0f - z) * _h + z * hidden_pre_value;
  }
}

template <>
__global__ void cal_cudnn_kernel<half>(const half* w_x_r,
                                       const half* w_x_z,
                                       const half* w_x_o,
                                       const half* w_h_r,
                                       const half* w_h_z,
                                       const half* w_h_o,
                                       int hidden_size,
                                       int batch_size,
                                       half* output,
                                       const half* hidden_pre) {
  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_id = thread_id / hidden_size;
  const int index = thread_id % hidden_size;
  if (index < hidden_size && batch_id < batch_size) {
    int w_base_index = batch_id * hidden_size * 3 + index;
    int h_base_index = batch_id * hidden_size + index;
    half hidden_pre_value = hidden_pre[h_base_index];
    half r = Sigmoid(w_x_r[w_base_index] + w_h_r[w_base_index]);
    half z = Sigmoid(w_x_z[w_base_index] + w_h_z[w_base_index]);
    half _h = Tanh(w_x_o[w_base_index] + w_h_o[w_base_index] * r);

    output[h_base_index] = __hadd(__hmul(__hsub(__float2half(1.0f), z), _h),
                                  __hmul(z, hidden_pre_value));
  }
}

template <typename T, PrecisionType PType>
void SearchGrnnCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& context = this->ctx_->template As<CUDAContext>();
  auto stream = context.exec_stream();

  const lite::Tensor* x = param.x;
  LoD offset_vec_vec = x->lod();
  std::vector<int> offset(offset_vec_vec[offset_vec_vec.size() - 1].size());
  for (size_t i = 0; i < offset_vec_vec[offset_vec_vec.size() - 1].size();
       ++i) {
    offset[i] = static_cast<int>(offset_vec_vec[offset_vec_vec.size() - 1][i]);
  }
  const T* x_data = x->data<T>();
  lite::Tensor* dout = param.out;
  std::vector<int64_t> out_dims_vec{x->dims()[0], param.num_hidden};
  dout->Resize(out_dims_vec);
  T* dout_data = dout->mutable_data<T>(TARGET(kCUDA));
  auto* wi = &_wi;
  auto* wh = &_wh;

  const T* weights_i2h = wi->data<T>();
  const T* weights_h2h = wh->data<T>();

  int batch_size = offset.size() - 1;
  int seq_sum = x->dims()[0];
  bool is_batched = offset.size() > 2;
  int hidden_size = param.num_hidden;
  int word_size = param.num_input;
  int o_offset = 0;
  int r_offset = 1;
  int z_offset = 2;

  is_batched = _seq_util.get_sorted_map(offset, stream);
  std::vector<int> emit_offset_vec = _seq_util.get_emit_offset_vec();
  int emit_length = emit_offset_vec.size() - 1;

  if (is_batched) {
    std::vector<int64_t> seq_shape{1, 1, seq_sum, word_size};
    _temp_tensor_in.Resize(seq_shape);
    std::vector<int64_t> seq_out_shape{1, 1, seq_sum, hidden_size};
    _temp_tensor_out.Resize(seq_out_shape);
    _seq_util.seq_2_sorted_seq(x_data,
                               _temp_tensor_in.mutable_data<T>(TARGET(kCUDA)),
                               word_size,
                               stream);
    x_data = _temp_tensor_in.data<T>();
    dout_data = _temp_tensor_out.mutable_data<T>(TARGET(kCUDA));
  }

  std::vector<int64_t> shape_wx({seq_sum, 1, 3, hidden_size});
  _temp_wx.Resize(shape_wx);

  std::vector<int64_t> shape_wh({1, batch_size, 3, hidden_size});
  _temp_wh.Resize(shape_wh);

  gemm_impl_->init(false, false, seq_sum, 3 * hidden_size, word_size, &context);
  gemm_impl_->run(1.0f,
                  0.0f,
                  x_data,
                  weights_i2h,
                  _temp_wx.mutable_data<T>(TARGET(kCUDA)),
                  &context);

  std::vector<int64_t> shape_zero({batch_size * hidden_size});
  _temp_zero.Resize(shape_zero);

  TargetWrapperCuda::MemsetAsync(_temp_zero.mutable_data<T>(TARGET(kCUDA)),
                                 0,
                                 sizeof(T) * batch_size * hidden_size,
                                 stream);

  const T* h = _temp_zero.data<T>();
  for (int word_id = 0; word_id < emit_length; word_id++) {
    int real_word_id = word_id;
    int last_word_id = word_id - 1;
    int emit_word_id_start = emit_offset_vec[real_word_id];
    int emit_word_id_end = emit_offset_vec[real_word_id + 1];
    int emit_word_length = emit_word_id_end - emit_word_id_start;

    const T* hidden_in;
    T* hidden_out = dout_data + emit_offset_vec[real_word_id] * hidden_size;

    if (word_id == 0) {
      hidden_in = h;
    } else {
      hidden_in = dout_data + emit_offset_vec[last_word_id] * hidden_size;
    }

    T* w_x_r = _temp_wx.mutable_data<T>(TARGET(kCUDA)) +
               r_offset * hidden_size + emit_word_id_start * hidden_size * 3;
    T* w_x_z = _temp_wx.mutable_data<T>(TARGET(kCUDA)) +
               z_offset * hidden_size + emit_word_id_start * hidden_size * 3;
    T* w_x_o = _temp_wx.mutable_data<T>(TARGET(kCUDA)) +
               o_offset * hidden_size + emit_word_id_start * hidden_size * 3;

    T* w_h_r = _temp_wh.mutable_data<T>(TARGET(kCUDA)) + r_offset * hidden_size;
    T* w_h_z = _temp_wh.mutable_data<T>(TARGET(kCUDA)) + z_offset * hidden_size;
    T* w_h_o = _temp_wh.mutable_data<T>(TARGET(kCUDA)) + o_offset * hidden_size;
    gemm_impl_->init(
        false, false, emit_word_length, 3 * hidden_size, hidden_size, &context);
    gemm_impl_->run(1.0f,
                    0.0f,
                    hidden_in,
                    _temp_weights_h2h.data<T>(),
                    _temp_wh.mutable_data<T>(TARGET(kCUDA)),
                    &context);

    const T* w_o = weights_h2h;
    const int block_dim = 512;
    const int grid_dim =
        (emit_word_length * hidden_size + block_dim - 1) / block_dim;
    cal_cudnn_kernel<<<grid_dim, block_dim, 0, stream>>>(w_x_r,
                                                         w_x_z,
                                                         w_x_o,
                                                         w_h_r,
                                                         w_h_z,
                                                         w_h_o,
                                                         hidden_size,
                                                         emit_word_length,
                                                         hidden_out,
                                                         hidden_in);
  }

  if (is_batched) {
    _seq_util.sorted_seq_2_seq(_temp_tensor_out.data<T>(),
                               dout->mutable_data<T>(TARGET(kCUDA)),
                               hidden_size,
                               stream);
  }

  dout->set_lod(x->lod());
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using GrnnFp32 =
    paddle::lite::kernels::cuda::SearchGrnnCompute<float, PRECISION(kFloat)>;
using GrnnFp16 =
    paddle::lite::kernels::cuda::SearchGrnnCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(search_grnn, kCUDA, kFloat, kNCHW, GrnnFp32, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Wi",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Wh",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("tmp_buffer",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("idx_sorted_by_width",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("layout_input",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(search_grnn, kCUDA, kFP16, kNCHW, GrnnFp16, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Wi",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("Wh",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("tmp_buffer",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("idx_sorted_by_width",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("layout_input",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
