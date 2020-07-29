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

#include "lite/kernels/cuda/match_matrix_tensor_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/api/test_helper.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class MatchMatrixTest : public ::testing::Test {
 protected:
  MatchMatrixTest()
      : ix_(6),
        iy_(4),
        h_(2),
        dim_t_(2),
        x_shape_({ix_, h_}),
        w_shape_({h_, dim_t_, h_}),
        y_shape_({iy_, h_}),
        x_lod_({{0, 3, 6}}),
        y_lod_({{0, 3, 4}}) {
    int batch = y_lod_[0].size() - 1;
    int len_l = x_lod_[0][1] - x_lod_[0][0];
    for (size_t i = 1; i < x_lod_[0].size() - 1; i++) {
      int cur_len = x_lod_[0][i + 1] - x_lod_[0][i];
      CHECK_EQ(cur_len, len_l)
          << "each sequence of left matrix is the same length";
    }
    int max_len_r = 0;
    for (size_t i = 0; i < y_lod_[0].size() - 1; ++i) {
      int cur_len = y_lod_[0][i + 1] - y_lod_[0][i];
      max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
    }
    out_shape_.clear();
    out_shape_.push_back(batch);
    out_shape_.push_back(dim_t_);
    out_shape_.push_back(len_l);
    out_shape_.push_back(max_len_r);

    x_ref_.Resize(lite::DDim(x_shape_));
    x_gpu_.Resize(lite::DDim(x_shape_));
    x_ref_.set_lod(x_lod_);

    w_ref_.Resize(lite::DDim(w_shape_));
    w_gpu_.Resize(lite::DDim(w_shape_));

    y_gpu_.Resize(lite::DDim(y_shape_));
    y_ref_.Resize(lite::DDim(y_shape_));
    y_ref_.set_lod(y_lod_);

    auto x_ref_data = x_ref_.mutable_data<float>();
    auto w_ref_data = w_ref_.mutable_data<float>();
    auto y_ref_data = y_ref_.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < x_ref_.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i);
    }
    for (int64_t i = 0; i < w_ref_.numel(); i++) {
      w_ref_data[i] = static_cast<float>(i);
    }
    for (int64_t i = 0; i < y_ref_.numel(); i++) {
      y_ref_data[i] = static_cast<float>(i);
    }

    out_ref_.Resize(lite::DDim(out_shape_));
    out_cpu_.Resize(lite::DDim(out_shape_));
    RunBaseLine(&x_ref_, &w_ref_, &y_ref_, &out_ref_);

    InitParamAndContext();
  }

  void InitParamAndContext() {
    ctx_.reset(new KernelContext);
    cudaStreamCreate(&stream_);
    auto& context = ctx_->As<CUDAContext>();
    context.SetExecStream(stream_);
    param_.x = &x_gpu_;
    param_.w = &w_gpu_;
    param_.y = &y_gpu_;
    param_.dim_t = dim_t_;
    param_.out = &out_gpu_;
    param_.tmp = &out_tmp_;
  }

  void InitFloatInput() {
    x_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref_.data<float>(),
                                                    x_gpu_.dims());
    y_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(y_ref_.data<float>(),
                                                    y_gpu_.dims());
    x_gpu_.set_lod(x_ref_.lod());
    y_gpu_.set_lod(y_ref_.lod());
    w_gpu_.Assign<float, lite::DDim, TARGET(kCUDA)>(w_ref_.data<float>(),
                                                    w_gpu_.dims());
  }

  void InitHalfInput() {
    x_half_.Resize(lite::DDim(x_shape_));
    auto x_half_data = x_half_.mutable_data<half>();
    for (int64_t i = 0; i < x_half_.numel(); i++) {
      x_half_data[i] = half(lite::float16(x_ref_.data<float>()[i]));
    }
    x_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu_.dims());
    x_gpu_.set_lod(x_ref_.lod());
    y_half_.Resize(lite::DDim(y_shape_));
    auto y_half_data = y_half_.mutable_data<half>();
    for (int64_t i = 0; i < y_half_.numel(); i++) {
      y_half_data[i] = half(lite::float16(y_ref_.data<float>()[i]));
    }
    y_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(y_half_data, y_gpu_.dims());
    y_gpu_.set_lod(y_ref_.lod());
    w_half_.Resize(w_ref_.dims());
    auto w_half_data = w_half_.mutable_data<half>();
    for (int64_t i = 0; i < w_half_.numel(); i++) {
      w_half_data[i] = half(lite::float16(w_ref_.data<float>()[i]));
    }
    w_gpu_.Assign<half, lite::DDim, TARGET(kCUDA)>(w_half_data, w_gpu_.dims());
  }

  void RunBaseLine(const lite::Tensor* x,
                   const lite::Tensor* w,
                   const lite::Tensor* b,
                   lite::Tensor* out) {
    std::vector<float> ref_results = {5,  23, 41, 17,  75,  133, 29,  127, 225,
                                      7,  33, 59, 27,  125, 223, 47,  217, 387,
                                      59, 0,  0,  191, 0,   0,   323, 0,   0,
                                      85, 0,  0,  321, 0,   0,   557, 0,   0};
    for (size_t i = 0; i < ref_results.size(); ++i) {
      out->mutable_data<float>()[i] = ref_results[i];
    }
  }

  int ix_, iy_, h_, dim_t_;
  std::vector<int64_t> x_shape_, w_shape_, y_shape_, out_shape_;
  LoD x_lod_, y_lod_;
  lite::Tensor x_ref_, w_ref_, y_ref_, out_ref_;
  lite::Tensor x_gpu_, w_gpu_, y_gpu_;
  lite::Tensor x_half_, y_half_, w_half_;
  lite::Tensor out_cpu_, out_gpu_, out_tmp_;

  operators::MatchMatrixTensorParam param_;
  std::unique_ptr<KernelContext> ctx_;
  cudaStream_t stream_;
};

TEST_F(MatchMatrixTest, TestFP32) {
  InitFloatInput();
  MatchMatrixTensorCompute<float, PRECISION(kFloat)> match_matrix_kernel;
  match_matrix_kernel.SetParam(param_);
  match_matrix_kernel.SetContext(std::move(ctx_));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    match_matrix_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  match_matrix_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    match_matrix_kernel.Run();
  }
  cudaDeviceSynchronize();
  auto duration = (GetCurrentUS() - start) / 1000.0;
  LOG(INFO) << "fp32, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";

  CopySync<TARGET(kCUDA)>(out_cpu_.mutable_data<float>(),
                          out_gpu_.data<float>(),
                          sizeof(float) * out_gpu_.numel(),
                          IoDirection::DtoH);
  for (int i = 0; i < out_gpu_.numel(); ++i) {
    EXPECT_NEAR(out_cpu_.data<float>()[i], out_ref_.data<float>()[i], 5e-4);
  }
}

// TEST_F(MatchMatrixTest, TestFP16) {
//   InitHalfInput();
//   MatchMatrixTensorCompute<half, PRECISION(kFP16)> match_matrix_kernel;
//   match_matrix_kernel.SetParam(param_);
//   match_matrix_kernel.SetContext(std::move(ctx_));

//   for (int i = 0; i < FLAGS_warmup; ++i) {
//     match_matrix_kernel.Launch();
//     cudaDeviceSynchronize();
//   }

//   auto start = GetCurrentUS();
//   match_matrix_kernel.PrepareForRun();
//   for (int i = 0; i < FLAGS_repeats; ++i) {
//     match_matrix_kernel.Run();
//   }
//   cudaDeviceSynchronize();
//   auto duration = (GetCurrentUS() - start) / 1000.0;
//   LOG(INFO) << "fp16, warmup: " << FLAGS_warmup
//             << ", repeats: " << FLAGS_repeats << ", spend "
//             << duration / FLAGS_repeats << " ms in average.";

//   const half* out_gpu_data = Out_gpu.data<half>();
//   half* out_cpu_data = Out_cpu.mutable_data<half>();
//   CopySync<TARGET(kCUDA)>(out_cpu_data,
//                           out_gpu_data,
//                           sizeof(half) * Out_gpu.numel(),
//                           IoDirection::DtoH);

//   for (int i = 0; i < Out_cpu.numel(); ++i) {
//     float res = static_cast<float>(lite::float16(out_cpu_data[i]));
//     float ref = Out_ref.data<float>()[i];
//     EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-2);
//   }
// }

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
