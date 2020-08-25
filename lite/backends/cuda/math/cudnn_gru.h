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

#pragma once
#include <cudnn.h>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/math/cudnn_helper.h"
#include "lite/backends/cuda/math/sequence_helper.h"
#include "lite/backends/cuda/math/transpose.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename T, PrecisionType Ptype>
class CudnnGRU {
 public:
  CudnnGRU()
      : handle_(nullptr),
        dropout_desc_(nullptr),
        rnn_desc_(nullptr),
        hx_desc_(nullptr),
        cx_desc_(nullptr),
        hy_desc_(nullptr),
        cy_desc_(nullptr),
        w_desc_(nullptr) {}

  ~CudnnGRU() {
    if (dropout_desc_) {
      CUDNN_CHECK(cudnnDestroyDropoutDescriptor(dropout_desc_));
    }
    if (rnn_desc_) {
      CUDNN_CHECK(cudnnDestroyRNNDescriptor(rnn_desc_));
    }
    if (hx_desc_) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(hx_desc_));
    }
    if (cx_desc_) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(cx_desc_));
    }
    if (hy_desc_) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(hy_desc_));
    }
    if (cy_desc_) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(cy_desc_));
    }
    if (w_desc_) {
      CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc_));
    }
    if (handle_) {
      CUDNN_CHECK(cudnnDestroy(handle_));
    }
  }

  bool Init(const lite::operators::GRUParam& param,
            Context<TARGET(kCUDA)>* ctx);

  bool Create(const lite::operators::GRUParam& param,
              Context<TARGET(kCUDA)>* ctx);

  bool Run(const lite::operators::GRUParam& param);

 private:
  int GetGruParams(const lite::operators::GRUParam& param);
  void SetGruParams(const lite::operators::GRUParam& param);
  void TransLiteWeights2CudnnWeights(const lite::operators::GRUParam& param);

 private:
  cudaStream_t stream_;
  cudnnHandle_t handle_;
  cudnnDropoutDescriptor_t dropout_desc_;
  cudnnRNNDescriptor_t rnn_desc_;

  // gate desc
  cudnnTensorDescriptor_t hx_desc_;
  cudnnTensorDescriptor_t cx_desc_;
  cudnnTensorDescriptor_t hy_desc_;
  cudnnTensorDescriptor_t cy_desc_;
  cudnnFilterDescriptor_t w_desc_;
  const int cudnn_gru_weights_layernum_ = 6;

  // input and output descs
  std::unique_ptr<cudnn::TensorDescriptors<T>> x_descs_;
  std::unique_ptr<cudnn::TensorDescriptors<T>> y_descs_;

  // workspace for cudnn
  const size_t workspace_limit_bytes_ = 4 * 1024 * 1024;
  size_t workspace_size_in_bytes_;
  lite::Tensor workspace_tensor_;

  lite::Tensor inner_weight_;
  lite::Tensor inner_weight_i2h_;
  lite::Tensor inner_weight_h2h_;
  lite::Tensor tmp_tensor_in_;
  lite::Tensor tmp_tensor_out_;
  std::vector<cudnn::ParamsRegion> inner_weight_region_;
  std::vector<cudnn::ParamsRegion> inner_bias_region_;

  lite::cuda::math::Transpose<T> trans_;
  SeqSortedseqTranseUtil seq_utils_;

  int word_size_;
  int hidden_size_;
  bool use_tensor_core_{true};
};

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
