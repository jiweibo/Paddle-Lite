// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/cuda/math/cudnn_gru.h"

#include "lite/backends/cuda/target_wrapper.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename T, PrecisionType Ptype>
void CudnnGRU<T, Ptype>::TransLiteWeights2CudnnWeights(
    const lite::operators::GRUParam& param) {
  inner_weight_i2h_.Resize({3, hidden_size_, word_size_});

  trans_.transpose(inner_weight_i2h_.mutable_data<T>(TARGET(kCUDA)),
                   param.weight_i2h->data<T>(),
                   {word_size_, 3, hidden_size_},
                   {1, 2, 0},
                   &stream_);

  inner_weight_h2h_.Resize({3, hidden_size_, hidden_size_});
  trans_.transpose(inner_weight_h2h_.mutable_data<T>(TARGET(kCUDA)),
                   param.weight->data<T>(),
                   {hidden_size_, 2, hidden_size_},
                   {1, 2, 0},
                   &stream_);
  // lite::TargetWrapperCuda::MemcpyAsync(
  //     reinterpret_cast<void*>(inner_weight_h2h_.mutable_data<T>(TARGET(kCUDA))
  //     +
  //                             hidden_size_ * hidden_size_ * 2),
  //     param.weight->data<T>() + 2 * hidden_size_ * hidden_size_,
  //     hidden_size_ * hidden_size_ * sizeof(T),
  //     IoDirection::DtoD,
  //     stream_);
  trans_.transpose(inner_weight_h2h_.mutable_data<T>(TARGET(kCUDA)) +
                       hidden_size_ * hidden_size_ * 2,
                   param.weight->data<T>() + 2 * hidden_size_ * hidden_size_,
                   {hidden_size_, hidden_size_},
                   {1, 0},
                   &stream_);
}

template <typename T, PrecisionType Ptype>
int CudnnGRU<T, Ptype>::GetGruParams(const lite::operators::GRUParam& param) {
  int sum_size_of_weights_and_bias = 0;
  cudnnFilterDescriptor_t param_desc_handle = nullptr;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&param_desc_handle));
  /**
   * gru in rnn has 6 bias layer
   */
  int region_count_of_layer = cudnn_gru_weights_layernum_;
  // TODO(wilber): num_layers > 1
  const int num_layers = 1;
  // LOG(INFO) << "weight addr is "
  //           << inner_weight_.mutable_data<T>(TARGET(kCUDA));
  for (int layer = 0; layer < num_layers; ++layer) {
    for (int region = 0; region < region_count_of_layer; ++region) {
      for (int trigger = 0; trigger < 2; ++trigger) {
        void* offset = nullptr;
        // LOG(INFO) << "region is " << region;
        if (trigger == 0) { /* weights */
          CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
              handle_,
              rnn_desc_,
              layer,
              x_descs_->descs()[0],
              w_desc_,
              inner_weight_.mutable_data<T>(TARGET(kCUDA)),
              region,
              param_desc_handle,
              &offset));
          // LOG(INFO) << "offset addr is " << offset;
        } else { /* bias */
          CUDNN_CHECK(cudnnGetRNNLinLayerBiasParams(
              handle_,
              rnn_desc_,
              layer,
              x_descs_->descs()[0],
              w_desc_,
              inner_weight_.mutable_data<T>(TARGET(kCUDA)),
              region,
              param_desc_handle,
              &offset));
        }

        int dims[] = {1, 1, 1};
        cudnnDataType_t data_type;
        cudnnTensorFormat_t tensor_format;
        int nbDims;
        CUDNN_CHECK(cudnnGetFilterNdDescriptor(param_desc_handle,
                                               sizeof(dims) / sizeof(dims[0]),
                                               &data_type,
                                               &tensor_format,
                                               &nbDims,
                                               dims));
        size_t size = dims[0] * dims[1] * dims[2] * sizeof(T);
        sum_size_of_weights_and_bias += size;
        // LOG(INFO) << "size is " << size;
        auto rg = cudnn::ParamsRegion(offset, size);
        if (trigger == 0) {
          inner_weight_region_.push_back(rg);
        } else {
          inner_bias_region_.push_back(rg);
        }
      }
    }
  }
  return sum_size_of_weights_and_bias;
}

template <typename T, PrecisionType Ptype>
void CudnnGRU<T, Ptype>::SetGruParams(const lite::operators::GRUParam& param) {
  const T* w_i2h_ptr = inner_weight_i2h_.data<T>();
  const T* w_h2h_ptr = inner_weight_h2h_.data<T>();

  const T* i2h_z = w_i2h_ptr;                                    // update gate
  const T* i2h_r = w_i2h_ptr + 1 * word_size_ * hidden_size_;    // reset gate
  const T* i2h_o = w_i2h_ptr + 2 * word_size_ * hidden_size_;    // memory gate
  const T* h2h_z = w_h2h_ptr;                                    // update gate
  const T* h2h_r = w_h2h_ptr + 1 * hidden_size_ * hidden_size_;  // reset gate
  const T* h2h_o = w_h2h_ptr + 2 * hidden_size_ * hidden_size_;  // memory gate

  const T* h_z = nullptr;
  const T* h_r = nullptr;
  const T* h_o = nullptr;

  if (param.bias != nullptr) {
    h_z = param.bias->data<T>();
    h_r = h_z + 1 * hidden_size_;
    h_o = h_z + 2 * hidden_size_;
  }

  const T* cudnnW[] = {i2h_r, i2h_z, i2h_o, h2h_r, h2h_z, h2h_o};
  const T* cudnnB[] = {h_r, h_z, h_o, nullptr, nullptr, nullptr};

  for (int i = 0; i < cudnn_gru_weights_layernum_; ++i) {
    cudnn::ParamsRegion& region = inner_weight_region_[i];
    lite::TargetWrapperCuda::MemcpyAsync(
        reinterpret_cast<void*>(region.offset_),
        cudnnW[i],
        region.size_,
        IoDirection::DtoD,
        stream_);

    cudnn::ParamsRegion& region_b = inner_bias_region_[i];
    if (cudnnB[i]) {
      lite::TargetWrapperCuda::MemcpyAsync(
          reinterpret_cast<void*>(region_b.offset_),
          cudnnB[i],
          region_b.size_,
          IoDirection::DtoD,
          stream_);
    } else {
      lite::TargetWrapperCuda::MemsetAsync(
          reinterpret_cast<void*>(region_b.offset_),
          0,
          region_b.size_,
          stream_);
    }
  }
}

template <typename T, PrecisionType Ptype>
bool CudnnGRU<T, Ptype>::Init(const operators::GRUParam& param,
                              Context<TARGET(kCUDA)>* ctx) {
  this->stream_ = ctx->exec_stream();
  CUDNN_CHECK(cudnnCreate(&this->handle_));
  CUDNN_CHECK(cudnnSetStream(this->handle_, this->stream_));

  hidden_size_ = param.weight->dims()[0];
  word_size_ = param.weight_i2h->dims()[0];

  CUDNN_CHECK(cudnnCreateDropoutDescriptor(&dropout_desc_));
  CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc_));
  const int num_layers = 1;
  CUDNN_CHECK(cudnnSetRNNDescriptor_v6(handle_,
                                       rnn_desc_,
                                       hidden_size_,
                                       num_layers,
                                       dropout_desc_,
                                       CUDNN_LINEAR_INPUT,
                                       CUDNN_UNIDIRECTIONAL,
                                       CUDNN_GRU,
                                       CUDNN_RNN_ALGO_STANDARD,
                                       cudnn::cudnnTypeWrapper<T>::type));
  // TODO(wilber): bias mode.
  if (param.bias) {
    CUDNN_CHECK(cudnnSetRNNBiasMode(rnn_desc_, CUDNN_RNN_SINGLE_INP_BIAS));
  } else {
    CUDNN_CHECK(cudnnSetRNNBiasMode(rnn_desc_, CUDNN_RNN_NO_BIAS));
  }
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&hx_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&cx_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&hy_desc_));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&cy_desc_));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc_));

  x_descs_.reset(new cudnn::TensorDescriptors<T>(
      1, {{1, word_size_, 1}}, {{word_size_, 1, 1}}));

  size_t weights_size;
  CUDNN_CHECK(cudnnGetRNNParamsSize(handle_,
                                    rnn_desc_,
                                    x_descs_->descs()[0],
                                    &weights_size,
                                    cudnn::cudnnTypeWrapper<T>::type));
  const int dims[] = {static_cast<int>(weights_size / sizeof(T)), 1, 1};
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(
      w_desc_, cudnn::cudnnTypeWrapper<T>::type, CUDNN_TENSOR_NCHW, 3, dims));
  inner_weight_.Resize({static_cast<int>(weights_size / sizeof(T))});

  TransLiteWeights2CudnnWeights(param);

  int sum_size_of_w = GetGruParams(param);
  // LOG(INFO) << "weights size is " << sum_size_of_w;
  CHECK_EQ(sum_size_of_w, weights_size)
      << "Compute param sum length must equal to cudnn api get.";

  SetGruParams(param);

  return Create(param, ctx);
}

template <typename T, PrecisionType Ptype>
bool CudnnGRU<T, Ptype>::Create(const operators::GRUParam& param,
                                Context<TARGET(kCUDA)>* ctx) {
  return true;
}

template <typename T, PrecisionType Ptype>
bool CudnnGRU<T, Ptype>::Run(const operators::GRUParam& param) {
  auto* input = param.input;
  lite::Tensor* h0{nullptr};
  if (param.h0) {
    h0 = const_cast<lite::Tensor*>(param.h0);
  }

  auto* in_data = input->template data<T>();
  lite::Tensor* hidden = param.hidden;
  auto* out_data = hidden->template mutable_data<T>(TARGET(kCUDA));

  param.hidden->set_lod(input->lod());
  std::vector<int> offset(input->lod()[0].size());
  for (size_t i = 0; i < input->lod()[0].size(); ++i) {
    offset[i] = input->lod()[0][i];
  }
  int max_batch_size = offset.size() - 1;
  bool need_process = seq_utils_.GetSortedMap(offset, stream_);
  auto offset_after_sort = seq_utils_.GetEmitOffsetVec();
  int max_seq_len = offset_after_sort.size() - 1;

  std::vector<std::vector<int>> xdim(max_seq_len);
  std::vector<std::vector<int>> xstride(max_seq_len);
  std::vector<std::vector<int>> ydim(max_seq_len);
  std::vector<std::vector<int>> ystride(max_seq_len);

  const int num_direction = 1;
  const int num_layers = 1;
  for (int i = 0; i < max_seq_len; ++i) {
    int length = offset_after_sort[i + 1] - offset_after_sort[i];
    xdim[i] = {length, word_size_, 1};
    xstride[i] = {word_size_, 1, 1};
    ydim[i] = {length, hidden_size_ * num_direction, 1};
    ystride[i] = {hidden_size_ * num_direction, 1, 1};
  }
  x_descs_.reset(new cudnn::TensorDescriptors<T>(max_seq_len, xdim, xstride));
  y_descs_.reset(new cudnn::TensorDescriptors<T>(max_seq_len, ydim, ystride));

  int dim[] = {num_layers * num_direction, max_batch_size, hidden_size_};
  int stride[] = {max_batch_size * hidden_size_, hidden_size_, 1};
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      hx_desc_, cudnn::cudnnTypeWrapper<T>::type, 3, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      cx_desc_, cudnn::cudnnTypeWrapper<T>::type, 3, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      hy_desc_, cudnn::cudnnTypeWrapper<T>::type, 3, dim, stride));
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      cy_desc_, cudnn::cudnnTypeWrapper<T>::type, 3, dim, stride));

  CUDNN_CHECK(cudnnGetRNNWorkspaceSize(handle_,
                                       rnn_desc_,
                                       max_seq_len,
                                       x_descs_->descs(),
                                       &workspace_size_in_bytes_));
  workspace_tensor_.Resize(
      {static_cast<int>(workspace_size_in_bytes_ / sizeof(T))});

  if (need_process) {
    tmp_tensor_in_.Resize(param.input->dims());
    tmp_tensor_out_.Resize(param.hidden->dims());
    auto* tmp_in_data = tmp_tensor_in_.mutable_data<T>(TARGET(kCUDA));
    auto* tmp_out_data = tmp_tensor_out_.mutable_data<T>(TARGET(kCUDA));
    seq_utils_.Seq2SortedSeq(in_data, tmp_in_data, word_size_, stream_);

    CUDNN_CHECK(cudnnRNNForwardInference(
        handle_,
        rnn_desc_,
        x_descs_->size(),
        x_descs_->descs(),
        tmp_in_data,
        hx_desc_,
        h0,
        cx_desc_,
        nullptr,
        w_desc_,
        inner_weight_.data<T>(),
        y_descs_->descs(),
        tmp_out_data,
        hy_desc_,
        nullptr,
        cy_desc_,
        nullptr,
        workspace_tensor_.mutable_data<T>(TARGET(kCUDA)),
        workspace_size_in_bytes_));
    seq_utils_.SortedSeq2Seq(tmp_out_data, out_data, hidden_size_, stream_);
  } else {
    CUDNN_CHECK(cudnnRNNForwardInference(
        handle_,
        rnn_desc_,
        x_descs_->size(),
        x_descs_->descs(),
        in_data,
        hx_desc_,
        h0,
        cx_desc_,
        nullptr,
        w_desc_,
        inner_weight_.data<T>(),
        y_descs_->descs(),
        out_data,
        hy_desc_,
        nullptr,
        cy_desc_,
        nullptr,
        workspace_tensor_.mutable_data<T>(TARGET(kCUDA)),
        workspace_size_in_bytes_));
  }
  return true;
}

template class CudnnGRU<float, PRECISION(kFloat)>;
template class CudnnGRU<half, PRECISION(kFP16)>;

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
