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

#include "lite/kernels/cuda/mul_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/api/test_helper.h"
#include "lite/utils/float16.h"

DEFINE_int32(m, 16, "m");
DEFINE_int32(n, 16, "n");
DEFINE_int32(k, 16, "k");

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class MulTest : public ::testing::Test {
 protected:
  MulTest()
      : m(FLAGS_m), k(FLAGS_k), n(FLAGS_n), 
        x_shape({m, k}), y_shape({k, n}), out_shape({m, n}) {
    X_gpu.Resize(lite::DDim(x_shape));
    X_ref.Resize(lite::DDim(x_shape));

    Y_gpu.Resize(lite::DDim(y_shape));
    Y_ref.Resize(lite::DDim(y_shape));

    auto x_ref_data = X_ref.mutable_data<float>();
    auto y_ref_data = Y_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < Y_ref.numel(); i++) {
      y_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_cpu.Resize(lite::DDim(out_shape));
    Out_gpu.Resize(lite::DDim(out_shape));
    fc_cpu_base(&X_ref, &Y_ref, &Out_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
    param.x = &X_gpu;
    param.y = &Y_gpu;
    param.output = &Out_gpu;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
    Y_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(Y_ref.data<float>(),
                                                   Y_gpu.dims());
  }

  void half_data_init() {
    X_half.Resize(lite::DDim(X_ref.dims()));
    auto x_half_data = X_half.mutable_data<half>();
    for (int64_t i = 0; i < X_half.numel(); i++) {
      x_half_data[i] = half(lite::float16(X_ref.data<float>()[i]));
    }
    X_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, X_gpu.dims());
    Y_half.Resize(Y_ref.dims());
    auto y_half_data = Y_half.mutable_data<half>();
    for (int64_t i = 0; i < Y_half.numel(); i++) {
      y_half_data[i] = half(lite::float16(Y_ref.data<float>()[i]));
    }
    Y_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(y_half_data, Y_gpu.dims());
  }

  void fc_cpu_base(const lite::Tensor* X,
                   const lite::Tensor* W,
                   lite::Tensor* Out) {
    const float* data_in = X->data<float>();
    const float* weights = W->data<float>();
    float* data_out = Out->mutable_data<float>();
    int out_rows = X->dims()[0];
    int in_cols = X->numel() / out_rows;
    int out_cols = W->numel() / in_cols;
    int index_out;
    for (int i = 0; i < out_rows; i++) {
      for (int j = 0; j < out_cols; j++) {
        index_out = i * out_cols + j;
        data_out[index_out] = 0;
        for (int k = 0; k < in_cols; k++) {
          data_out[index_out] +=
              data_in[i * in_cols + k] * weights[k * out_cols + j];
        }
      }
    }
  }

  int m, k, n;
  std::vector<int64_t> x_shape, y_shape, out_shape;
  lite::Tensor X_ref, Y_ref, Out_ref;
  lite::Tensor X_gpu, Y_gpu;
  lite::Tensor X_half, Y_half;
  lite::Tensor Out_cpu, Out_gpu;

  operators::MulParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(MulTest, TestFP32) {
  float_data_init();
  MulCompute<float, PRECISION(kFloat)> mul_kernel;
  mul_kernel.SetParam(param);
  mul_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    mul_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  mul_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    mul_kernel.Run();
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
    float res = Out_cpu.data<float>()[i];
    float ref = Out_ref.data<float>()[i];
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-4);
  }
}

TEST_F(MulTest, TestFP16) {
  half_data_init();
  MulCompute<half, PRECISION(kFP16)> mul_kernel;
  mul_kernel.SetParam(param);
  mul_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    mul_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  mul_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    mul_kernel.Run();
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
