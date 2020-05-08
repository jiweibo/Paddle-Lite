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
#include <map>
#include <vector>
#include "lite/backends/cuda/math/elementwise.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/elementwise_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

inline DDim trim_trailing_singular_dims(const DDim& dims) {
  // Remove trailing dimensions of size 1 for y
  auto actual_dims_size = dims.size();
  for (; actual_dims_size != 0; --actual_dims_size) {
    if (dims[actual_dims_size - 1] != 1) break;
  }

  std::vector<int64_t> trim_dims;
  trim_dims.resize(actual_dims_size);
  for (int i = 0; i < actual_dims_size; ++i) {
    trim_dims[i] = dims[i];
  }
  if (trim_dims.size() == 0) {
    return DDim();
  }
  return DDim(trim_dims);
}

inline bool is_broadcast(const DDim& x_dims,
                         const DDim& y_dims,
                         int axis,
                         int* pre,
                         int* n,
                         int* post) {
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  DDim y_dim_trim = trim_trailing_singular_dims(y_dims);
  axis = (y_dim_trim.size() == 0) ? x_dims.size() : axis;
  if (x_dims.size() == y_dim_trim.size()) {
    return false;
  }
  *pre = 1;
  *n = 1;
  *post = 1;
  for (int i = 0; i < axis; ++i) {
    (*pre) *= x_dims[i];
  }
  for (int i = 0; i < y_dim_trim.size(); ++i) {
    CHECK_EQ(x_dims[i + axis], y_dim_trim[i])
        << "Broadcast dimension mismatch.";
    (*n) *= y_dim_trim[i];
  }
  for (int i = axis + y_dim_trim.size(); i < x_dims.size(); ++i) {
    (*post) *= x_dims[i];
  }
  return true;
}

#define ELEMENTWISE_COMPUTE(OP)                                    \
  auto& param = this->template Param<param_t>();                   \
  auto& ctx = this->ctx_->template As<CUDAContext>();              \
  auto stream = ctx.exec_stream();                                 \
  const lite::Tensor* x = param.X;                                 \
  const lite::Tensor* y = param.Y;                                 \
  lite::Tensor* out = param.Out;                                   \
  int axis = param.axis;                                           \
  auto* x_data = x->template data<T>();                            \
  auto* y_data = y->template data<T>();                            \
  auto out_data = out->template mutable_data<T>(TARGET(kCUDA));    \
  int pixel_num = x->numel();                                      \
  int pre = 1;                                                     \
  int n = pixel_num;                                               \
  int post = 1;                                                    \
  if (is_broadcast(x->dims(), y->dims(), axis, &pre, &n, &post)) { \
    lite::cuda::math::elementwise<T>(                              \
        x_data, y_data, out_data, pre, n, post, OP, stream);       \
  } else {                                                         \
    lite::cuda::math::elementwise<T>(                              \
        x_data, y_data, out_data, 1, pixel_num, 1, OP, stream);    \
  }

#define ELEMENTWISE_COMPUTE_ACT(OP)                                  \
  auto& param = this->Param<param_t>();                              \
  auto& ctx = this->ctx_->template As<CUDAContext>();                \
  auto stream = ctx.exec_stream();                                   \
  const lite::Tensor* x = param.X;                                   \
  const lite::Tensor* y = param.Y;                                   \
  lite::Tensor* out = param.Out;                                     \
  int axis = param.axis;                                             \
  auto* x_data = x->data<float>();                                   \
  auto* y_data = y->data<float>();                                   \
  auto out_data = out->mutable_data<float>(TARGET(kCUDA));           \
  int pixel_num = x->numel();                                        \
  int pre = 1;                                                       \
  int n = pixel_num;                                                 \
  int post = 1;                                                      \
  auto act = param.act_type;                                         \
  if (is_broadcast(x->dims(), y->dims(), axis, &pre, &n, &post)) {   \
    lite::cuda::math::elementwise_act(                               \
        x_data, y_data, out_data, pre, n, post, act, OP, stream);    \
  } else {                                                           \
    lite::cuda::math::elementwise_act(                               \
        x_data, y_data, out_data, 1, pixel_num, 1, act, OP, stream); \
  }

#define ELEMENTWISE_COMPUTE_NHWC(OP)                               \
  std::map<int, int> pos_map = {{0, 0}, {1, 3}, {2, 1}, {3, 2}};   \
  auto& param = this->template Param<param_t>();                   \
  auto& ctx = this->ctx_->template As<CUDAContext>();              \
  auto stream = ctx.exec_stream();                                 \
  const lite::Tensor* x = param.X;                                 \
  const lite::Tensor* y = param.Y;                                 \
  lite::Tensor* out = param.Out;                                   \
  int axis = param.axis;                                           \
  if (axis < 0) axis = x->dims().size() - y->dims().size();        \
  CHECK(axis >= 0) << "invalid axis of elementwise op";            \
  axis = pos_map[axis];                                            \
  auto* x_data = x->template data<T>();                            \
  auto* y_data = y->template data<T>();                            \
  auto out_data = out->mutable_data<T>(TARGET(kCUDA));             \
  int pixel_num = x->numel();                                      \
  int pre = 1;                                                     \
  int n = pixel_num;                                               \
  int post = 1;                                                    \
  if (is_broadcast(x->dims(), y->dims(), axis, &pre, &n, &post)) { \
    lite::cuda::math::elementwise<T>(                              \
        x_data, y_data, out_data, pre, n, post, OP, stream);       \
  } else {                                                         \
    lite::cuda::math::elementwise<T>(                              \
        x_data, y_data, out_data, 1, pixel_num, 1, OP, stream);    \
  }

#define ELEMENTWISE_COMPUTE_ACT_NHWC(OP)                             \
  std::map<int, int> pos_map = {{0, 0}, {1, 3}, {2, 1}, {3, 2}};     \
  auto& param = this->Param<param_t>();                              \
  auto& ctx = this->ctx_->template As<CUDAContext>();                \
  auto stream = ctx.exec_stream();                                   \
  const lite::Tensor* x = param.X;                                   \
  const lite::Tensor* y = param.Y;                                   \
  lite::Tensor* out = param.Out;                                     \
  int axis = param.axis;                                             \
  if (axis < 0) axis = x->dims().size() - y->dims().size();          \
  CHECK(axis >= 0) << "invalid axis of elementwise op";              \
  axis = pos_map[axis];                                              \
  auto* x_data = x->data<float>();                                   \
  auto* y_data = y->data<float>();                                   \
  auto out_data = out->mutable_data<float>(TARGET(kCUDA));           \
  int pixel_num = x->numel();                                        \
  int pre = 1;                                                       \
  int n = pixel_num;                                                 \
  int post = 1;                                                      \
  auto act = param.act_type;                                         \
  if (is_broadcast(x->dims(), y->dims(), axis, &pre, &n, &post)) {   \
    lite::cuda::math::elementwise_act(                               \
        x_data, y_data, out_data, pre, n, post, act, OP, stream);    \
  } else {                                                           \
    lite::cuda::math::elementwise_act(                               \
        x_data, y_data, out_data, 1, pixel_num, 1, act, OP, stream); \
  }

template <typename T, PrecisionType Ptype>
void ElementwiseAddCompute<T, Ptype>::Run() {
  ELEMENTWISE_COMPUTE(lite::cuda::math::BinaryOperation::kADD)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

template <typename T, PrecisionType Ptype>
void ElementwiseAddComputeNHWC<T, Ptype>::Run() {
  ELEMENTWISE_COMPUTE_NHWC(lite::cuda::math::BinaryOperation::kADD)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

template <typename T, PrecisionType Ptype>
void ElementwiseSubCompute<T, Ptype>::Run() {
  ELEMENTWISE_COMPUTE(lite::cuda::math::BinaryOperation::kSUB)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

template <typename T, PrecisionType Ptype>
void ElementwiseSubComputeNHWC<T, Ptype>::Run() {
  ELEMENTWISE_COMPUTE_NHWC(lite::cuda::math::BinaryOperation::kSUB)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

template <typename T, PrecisionType Ptype>
void ElementwiseMulCompute<T, Ptype>::Run() {
  ELEMENTWISE_COMPUTE(lite::cuda::math::BinaryOperation::kMUL)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

template <typename T, PrecisionType Ptype>
void ElementwiseMulComputeNHWC<T, Ptype>::Run() {
  ELEMENTWISE_COMPUTE_NHWC(lite::cuda::math::BinaryOperation::kMUL)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

void ElementwiseAddActivationCompute::Run() {
  ELEMENTWISE_COMPUTE_ACT(lite::cuda::math::BinaryOperation::kADD)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

void ElementwiseAddActivationComputeNHWC::Run() {
  ELEMENTWISE_COMPUTE_ACT_NHWC(lite::cuda::math::BinaryOperation::kADD)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

void ElementwiseSubActivationCompute::Run() {
  ELEMENTWISE_COMPUTE_ACT(lite::cuda::math::BinaryOperation::kSUB)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

void ElementwiseSubActivationComputeNHWC::Run() {
  ELEMENTWISE_COMPUTE_ACT_NHWC(lite::cuda::math::BinaryOperation::kSUB)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

void ElementwiseMulActivationCompute::Run() {
  ELEMENTWISE_COMPUTE_ACT(lite::cuda::math::BinaryOperation::kMUL)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

void ElementwiseMulActivationComputeNHWC::Run() {
  ELEMENTWISE_COMPUTE_ACT_NHWC(lite::cuda::math::BinaryOperation::kMUL)
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using ELT_ADD_FP32 =
    paddle::lite::kernels::cuda::ElementwiseAddCompute<float,
                                                       PRECISION(kFloat)>;
using ELT_SUB_FP32 =
    paddle::lite::kernels::cuda::ElementwiseSubCompute<float,
                                                       PRECISION(kFloat)>;
using ELT_MUL_FP32 =
    paddle::lite::kernels::cuda::ElementwiseMulCompute<float,
                                                       PRECISION(kFloat)>;

using ELT_ADD_NHWC_FP32 =
    paddle::lite::kernels::cuda::ElementwiseAddComputeNHWC<float,
                                                           PRECISION(kFloat)>;
using ELT_SUB_NHWC_FP32 =
    paddle::lite::kernels::cuda::ElementwiseSubComputeNHWC<float,
                                                           PRECISION(kFloat)>;
using ELT_MUL_NHWC_FP32 =
    paddle::lite::kernels::cuda::ElementwiseMulComputeNHWC<float,
                                                           PRECISION(kFloat)>;

REGISTER_LITE_KERNEL(elementwise_add, kCUDA, kFloat, kNCHW, ELT_ADD_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub, kCUDA, kFloat, kNCHW, ELT_SUB_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_add, kCUDA, kFloat, kNHWC, ELT_ADD_NHWC_FP32, nhwc_format)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_sub, kCUDA, kFloat, kNHWC, ELT_SUB_NHWC_FP32, nhwc_format)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul, kCUDA, kFloat, kNCHW, ELT_MUL_FP32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_mul, kCUDA, kFloat, kNHWC, ELT_MUL_NHWC_FP32, nhwc_format)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

using ELT_ADD_FP16 =
    paddle::lite::kernels::cuda::ElementwiseAddCompute<half, PRECISION(kFP16)>;
using ELT_SUB_FP16 =
    paddle::lite::kernels::cuda::ElementwiseSubCompute<half, PRECISION(kFP16)>;
using ELT_MUL_FP16 =
    paddle::lite::kernels::cuda::ElementwiseMulCompute<half, PRECISION(kFP16)>;

using ELT_ADD_NHWC_FP16 =
    paddle::lite::kernels::cuda::ElementwiseAddComputeNHWC<half,
                                                           PRECISION(kFP16)>;
using ELT_SUB_NHWC_FP16 =
    paddle::lite::kernels::cuda::ElementwiseSubComputeNHWC<half,
                                                           PRECISION(kFP16)>;
using ELT_MUL_NHWC_FP16 =
    paddle::lite::kernels::cuda::ElementwiseMulComputeNHWC<half,
                                                           PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(elementwise_add, kCUDA, kFP16, kNCHW, ELT_ADD_FP16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_sub, kCUDA, kFP16, kNCHW, ELT_SUB_FP16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_add, kCUDA, kFP16, kNHWC, ELT_ADD_NHWC_FP16, nhwc_format)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_sub, kCUDA, kFP16, kNHWC, ELT_SUB_NHWC_FP16, nhwc_format)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(elementwise_mul, kCUDA, kFP16, kNCHW, ELT_MUL_FP16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();

REGISTER_LITE_KERNEL(
    elementwise_mul, kCUDA, kFP16, kNHWC, ELT_MUL_NHWC_FP16, nhwc_format)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_add_activation,
    kCUDA,
    kFloat,
    kNCHW,
    paddle::lite::kernels::cuda::ElementwiseAddActivationCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_add_activation,
    kCUDA,
    kFloat,
    kNHWC,
    paddle::lite::kernels::cuda::ElementwiseAddActivationComputeNHWC,
    nhwc_format)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_sub_activation,
    kCUDA,
    kFloat,
    kNCHW,
    paddle::lite::kernels::cuda::ElementwiseSubActivationCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_sub_activation,
    kCUDA,
    kFloat,
    kNHWC,
    paddle::lite::kernels::cuda::ElementwiseSubActivationComputeNHWC,
    nhwc_format)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_mul_activation,
    kCUDA,
    kFloat,
    kNCHW,
    paddle::lite::kernels::cuda::ElementwiseMulActivationCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(
    fusion_elementwise_mul_activation,
    kCUDA,
    kFloat,
    kNHWC,
    paddle::lite::kernels::cuda::ElementwiseMulActivationComputeNHWC,
    nhwc_format)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
