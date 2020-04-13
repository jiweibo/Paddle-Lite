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

#pragma once

#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType PType>
class LookupTableCompute
    : public KernelLite<TARGET(kCUDA), PType, DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::LookupTableParam;

  void Run() override;
  void PrepareForRun() override;
  virtual ~LookupTableCompute() = default;

 private:
  const lite::Tensor* w_tensor_;
  lite::Tensor w_half_tensor_;
};

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
