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

#include "lite/backends/cuda/math/bias.h"
#include "lite/backends/cuda/math/type_trans.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/search_seq_fc_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType PType>
void SearchSeqFcCompute<T, PType>::PrepareForRun() {
  gemm_impl_.reset(new lite::cuda::math::Gemm<T, T>);
}

template <typename T, PrecisionType PType>
void SearchSeqFcCompute<T, PType>::Run() {
  auto& param = this->template Param<param_t>();
  auto& cuda_ctx = this->ctx_->template As<CUDAContext>();
  auto cuda_stream = cuda_ctx.exec_stream();

  auto x = param.x;
  auto w = param.w;
  auto b = param.b;
  auto out = param.out;
  auto out_size = param.out_size;
  const auto x_dims = x->dims();
  const auto w_dims = w->dims();
  const auto out_dims = out->dims();
  CHECK_EQ(x_dims.size(), 2) << "The Input(X) should be 2-D tensor.";
  CHECK_EQ(w_dims.size(), 2) << "W should be 2-D tensor.";
  CHECK_EQ(out_dims.size(), 2) << "The Output(Out) should be 2-D tensor.";
  CHECK_EQ(x_dims[1], w_dims[1]) << "Wrong shape: x_dims[1] != w_dims[1]";
  CHECK_EQ(w_dims[0], out_size) << "Wrong shape: w_dims[0] != out_size";
  CHECK_EQ(out_dims[0], x_dims[0]) << "Wrong shape: out_dims[0] != x_dims[0]";
  CHECK_EQ(out_dims[1], out_size) << "Wrong shape: out_dims[1] != out_size";
  int M = x_dims[0];
  int K = x_dims[1];
  int N = w_dims[0];
  auto x_data = x->template data<T>();
  auto w_data = w->template data<T>();
  auto out_data = out->template mutable_data<T>(TARGET(kCUDA));

  CHECK(gemm_impl_->init(false, true, M, N, K, &cuda_ctx));
  gemm_impl_->run(1.0f, 0.0f, x_data, w_data, out_data, &cuda_ctx);

  if (b != nullptr) {
    auto b_dims = b->dims();
    CHECK_EQ(b_dims.size(), 1) << "b should be 1-D tensor.";
    CHECK_EQ(b_dims[0], w_dims[0]) << "Wrong shape: b_dims[0] != w_dims[0]";
    auto b_data = b->template data<T>();
    int total_size = M * N;
    lite::cuda::math::add_bias(total_size, N, b_data, out_data, cuda_stream);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using SeqFCFp32 =
    paddle::lite::kernels::cuda::SearchSeqFcCompute<float, PRECISION(kFloat)>;
using SeqFCFp16 =
    paddle::lite::kernels::cuda::SearchSeqFcCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(search_seq_fc, kCUDA, kFloat, kNCHW, SeqFCFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(search_seq_fc, kCUDA, kFP16, kNCHW, SeqFCFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("b", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
