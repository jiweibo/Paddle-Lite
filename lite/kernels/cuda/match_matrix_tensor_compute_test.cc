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

using Tensor = lite::Tensor;

class MatchMatrixTest : public ::testing::Test {
 protected:
  MatchMatrixTest()
      : ix(6),
        iy(4),
        h(2),
        dim_t(2),
        x_shape({ix, h}),
        w_shape({h, dim_t, h}),
        y_shape({iy, h}),
        x_lod({{0, 3, 6}}),
        y_lod({{0, 3, 4}}) {
    int batch = y_lod[0].size() - 1;
    int len_l = x_lod[0][1] - x_lod[0][0];
    for (size_t i = 1; i < x_lod[0].size() - 1; i++) {
      int cur_len = x_lod[0][i + 1] - x_lod[0][i];
      CHECK_EQ(cur_len, len_l)
          << "each sequence of left matrix is the same length";
    }
    int max_len_r = 0;
    for (size_t i = 0; i < y_lod[0].size() - 1; ++i) {
      int cur_len = y_lod[0][i + 1] - y_lod[0][i];
      max_len_r = cur_len > max_len_r ? cur_len : max_len_r;
    }
    out_shape.clear();
    out_shape.push_back(batch);
    out_shape.push_back(dim_t);
    out_shape.push_back(len_l);
    out_shape.push_back(max_len_r);

    X_gpu.Resize(lite::DDim(x_shape));
    X_ref.Resize(lite::DDim(x_shape));
    X_ref.set_lod(x_lod);

    W_gpu.Resize(lite::DDim(w_shape));
    W_ref.Resize(lite::DDim(w_shape));

    Y_gpu.Resize(lite::DDim(y_shape));
    Y_ref.Resize(lite::DDim(y_shape));
    Y_ref.set_lod(y_lod);

    auto x_ref_data = X_ref.mutable_data<float>();
    auto w_ref_data = W_ref.mutable_data<float>();
    auto y_ref_data = Y_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i);
    }
    for (int64_t i = 0; i < W_ref.numel(); i++) {
      w_ref_data[i] = static_cast<float>(i);
    }
    for (int64_t i = 0; i < Y_ref.numel(); i++) {
      y_ref_data[i] = static_cast<float>(i);
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_cpu.Resize(lite::DDim(out_shape));
    cpu_ref(&X_ref, &W_ref, &Y_ref, &Out_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
    param.x = &X_gpu;
    param.w = &W_gpu;
    param.y = &Y_gpu;
    param.dim_t = dim_t;
    param.out = &Out_gpu;
    param.tmp = &Out_tmp;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
    Y_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(Y_ref.data<float>(),
                                                   Y_gpu.dims());
    X_gpu.set_lod(X_ref.lod());
    Y_gpu.set_lod(Y_ref.lod());
    W_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(W_ref.data<float>(),
                                                   W_gpu.dims());
  }

  void half_data_init() {
    X_half.Resize(lite::DDim(x_shape));
    auto x_half_data = X_half.mutable_data<half>();
    for (int64_t i = 0; i < X_half.numel(); i++) {
      x_half_data[i] = half(lite::float16(X_ref.data<float>()[i]));
    }
    X_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, X_gpu.dims());
    X_gpu.set_lod(X_ref.lod());
    Y_half.Resize(lite::DDim(y_shape));
    auto y_half_data = Y_half.mutable_data<half>();
    for (int64_t i = 0; i < Y_half.numel(); i++) {
      y_half_data[i] = half(lite::float16(Y_ref.data<float>()[i]));
    }
    Y_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(y_half_data, Y_gpu.dims());
    Y_gpu.set_lod(Y_ref.lod());
    W_half.Resize(W_ref.dims());
    auto w_half_data = W_half.mutable_data<half>();
    for (int64_t i = 0; i < W_half.numel(); i++) {
      w_half_data[i] = half(lite::float16(W_ref.data<float>()[i]));
    }
    W_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(w_half_data, W_gpu.dims());
  }

  void cpu_ref(const lite::Tensor* X,
               const lite::Tensor* W,
               const lite::Tensor* b,
               lite::Tensor* Out) {
    std::vector<float> ref_results = {5,  23, 41, 17,  75,  133, 29,  127, 225,
                                      7,  33, 59, 27,  125, 223, 47,  217, 387,
                                      59, 0,  0,  191, 0,   0,   323, 0,   0,
                                      85, 0,  0,  321, 0,   0,   557, 0,   0};
    for (size_t i = 0; i < ref_results.size(); ++i) {
      Out->mutable_data<float>()[i] = ref_results[i];
    }
  }

  int ix, iy, h, dim_t;
  std::vector<int64_t> x_shape, w_shape, y_shape, out_shape;
  LoD x_lod, y_lod;
  lite::Tensor X_ref, W_ref, Y_ref, Out_ref;
  lite::Tensor X_gpu, W_gpu, Y_gpu;
  lite::Tensor X_half, Y_half, W_half;
  lite::Tensor Out_cpu, Out_gpu, Out_tmp;

  operators::MatchMatrixTensorParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(MatchMatrixTest, TestFP32) {
  float_data_init();
  MatchMatrixTensorCompute<float, PRECISION(kFloat)> match_matrix_kernel;
  match_matrix_kernel.SetParam(param);
  match_matrix_kernel.SetContext(std::move(ctx));

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

  CopySync<TARGET(kCUDA)>(Out_cpu.mutable_data<float>(),
                          Out_gpu.data<float>(),
                          sizeof(float) * Out_gpu.numel(),
                          IoDirection::DtoH);
  for (int i = 0; i < Out_gpu.numel(); ++i) {
    EXPECT_NEAR(Out_cpu.data<float>()[i], Out_ref.data<float>()[i], 5e-4);
  }
}

TEST_F(MatchMatrixTest, TestFP16) {
  half_data_init();
  MatchMatrixTensorCompute<half, PRECISION(kFP16)> match_matrix_kernel;
  match_matrix_kernel.SetParam(param);
  match_matrix_kernel.SetContext(std::move(ctx));

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
  LOG(INFO) << "fp16, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";

  const half* out_gpu_data = Out_gpu.data<half>();
  half* out_cpu_data = Out_cpu.mutable_data<half>();
  CopySync<TARGET(kCUDA)>(out_cpu_data,
                          out_gpu_data,
                          sizeof(half) * Out_gpu.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < Out_cpu.numel(); ++i) {
    float res = static_cast<float>(lite::float16(out_cpu_data[i]));
    float ref = Out_ref.data<float>()[i];
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-2);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
