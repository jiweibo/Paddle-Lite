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

#include "lite/kernels/cuda/elementwise_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/api/test_helper.h"
#include "lite/backends/cuda/cuda_utils.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

using Tensor = lite::Tensor;

static void ElementwiseBroadcastRef(
    float* x, float* y, float* out, int pre, int n, int post) {
  for (int i = 0; i < pre * n * post; ++i) {
    int idx = (i / post) % n;
    out[i] = x[i] + y[idx];
  }
}

class EltWiseTest : public ::testing::Test {
 protected:
  EltWiseTest() : n_(1), c_(3), h_(512), w_(512), shape_info({n_, c_, h_, w_}) {
    X_gpu.Resize(lite::DDim(shape_info));
    X_ref.Resize(lite::DDim(shape_info));

    Y_gpu.Resize(lite::DDim(shape_info));
    Y_ref.Resize(lite::DDim(shape_info));

    auto x_ref_data = X_ref.mutable_data<float>();
    auto y_ref_data = Y_ref.mutable_data<float>();

    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 128) * 0.5f;
    }
    for (int64_t i = 0; i < Y_ref.numel(); i++) {
      y_ref_data[i] = static_cast<float>(i % 128) * 0.3f;
    }

    Out_ref.Resize(lite::DDim(shape_info));
    Out_cpu.Resize(lite::DDim(shape_info));
    Out_gpu.Resize(Out_ref.dims());
    cpu_ref(&X_ref, &Y_ref, &Out_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
    param.X = &X_gpu;
    param.Y = &Y_gpu;
    param.Out = &Out_gpu;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
    Y_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(Y_ref.data<float>(),
                                                   Y_gpu.dims());
  }

  void half_data_init() {
    X_half.Resize(lite::DDim(shape_info));
    auto x_half_data = X_half.mutable_data<half>();
    for (int64_t i = 0; i < X_half.numel(); i++) {
      x_half_data[i] = half(lite::float16(X_ref.data<float>()[i]));
    }
    X_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, X_gpu.dims());
    Y_half.Resize(lite::DDim(shape_info));
    auto y_half_data = Y_half.mutable_data<half>();
    for (int64_t i = 0; i < Y_half.numel(); i++) {
      y_half_data[i] = half(lite::float16(Y_ref.data<float>()[i]));
    }
    Y_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(y_half_data, Y_gpu.dims());
  }

  void cpu_ref(const lite::Tensor* X,
               const lite::Tensor* Y,
               lite::Tensor* Out) {
    for (int64_t i = 0; i < X->numel(); ++i) {
      Out->mutable_data<float>()[i] = X->data<float>()[i] + Y->data<float>()[i];
    }
  }

  int n_, c_, h_, w_;
  std::vector<int64_t> shape_info;
  lite::Tensor X_ref, Y_ref, Out_ref;
  lite::Tensor X_gpu, Y_gpu, Out_gpu;
  lite::Tensor X_half, Y_half;
  lite::Tensor Out_cpu;

  operators::ElementwiseParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(EltWiseTest, TestFP32) {
  float_data_init();
  ElementwiseAddCompute<float, PRECISION(kFloat)> kernel;
  kernel.SetParam(param);
  kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    kernel.Run();
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
    EXPECT_NEAR(Out_cpu.data<float>()[i], Out_ref.data<float>()[i], 1e-5);
  }
}

TEST_F(EltWiseTest, TestFP16) {
  half_data_init();
  ElementwiseAddCompute<half, PRECISION(kFP16)> kernel;
  kernel.SetParam(param);
  kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    kernel.Run();
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

TEST(elementwise_add, bias) {
  ElementwiseAddCompute<float, PRECISION(kFloat)> elementwise_add_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ElementwiseParam param;
  Tensor x, y, out;
  Tensor x_cpu, y_cpu, out_cpu;
  Tensor x_ref, y_ref, out_ref;

  const int n = 1;
  const int c = 3;
  const int h = 2000;
  const int w = 2000;

  x.Resize({n, c, h, w});
  y.Resize({c, 1, 1});
  out.Resize({n, c, h, w});
  x_cpu.Resize({n, c, h, w});
  y_cpu.Resize({c, 1, 1});
  out_cpu.Resize({n, c, h, w});
  x_ref.Resize({n, c, h, w});
  y_ref.Resize({c, 1, 1});
  out_ref.Resize({n, c, h, w});

  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));

  auto* x_cpu_data = x_cpu.mutable_data<float>();
  auto* y_cpu_data = y_cpu.mutable_data<float>();
  auto* out_cpu_data = out_cpu.mutable_data<float>();

  auto* x_ref_data = x_ref.mutable_data<float>();
  auto* y_ref_data = y_ref.mutable_data<float>();
  auto* out_ref_data = out_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i + 5.0;
    x_ref_data[i] = i + 5.0;
  }
  for (int i = 0; i < y_cpu.numel(); ++i) {
    y_cpu_data[i] = i - 5.0;
    y_ref_data[i] = i - 5.0;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  y.Assign<float, lite::DDim, TARGET(kCUDA)>(y_cpu_data, y_cpu.dims());

  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.axis = -1;
  elementwise_add_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  elementwise_add_kernel.SetContext(std::move(ctx));
  elementwise_add_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  ElementwiseBroadcastRef(x_ref_data, y_ref_data, out_ref_data, n, c, h * w);
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

TEST(elementwise_add_nhwc, bias) {
  ElementwiseAddComputeNHWC<float, PRECISION(kFloat)> elementwise_add_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();

  operators::ElementwiseParam param;
  Tensor x, y, out;
  Tensor x_cpu, y_cpu, out_cpu;
  Tensor x_ref, y_ref, out_ref;

  const int n = 1;
  const int c = 3;
  const int h = 2000;
  const int w = 2000;

  x.Resize({n, h, w, c});
  y.Resize({c, 1, 1});
  out.Resize({n, h, w, c});
  x_cpu.Resize({n, h, w, c});
  y_cpu.Resize({c, 1, 1});
  out_cpu.Resize({n, h, w, c});
  x_ref.Resize({n, h, w, c});
  y_ref.Resize({c, 1, 1});
  out_ref.Resize({n, h, w, c});

  auto* out_data = out.mutable_data<float>(TARGET(kCUDA));

  auto* x_cpu_data = x_cpu.mutable_data<float>();
  auto* y_cpu_data = y_cpu.mutable_data<float>();
  auto* out_cpu_data = out_cpu.mutable_data<float>();

  auto* x_ref_data = x_ref.mutable_data<float>();
  auto* y_ref_data = y_ref.mutable_data<float>();
  auto* out_ref_data = out_ref.mutable_data<float>();

  for (int i = 0; i < x_cpu.numel(); ++i) {
    x_cpu_data[i] = i + 5.0;
    x_ref_data[i] = i + 5.0;
  }
  for (int i = 0; i < y_cpu.numel(); ++i) {
    y_cpu_data[i] = i - 5.0;
    y_ref_data[i] = i - 5.0;
  }

  x.Assign<float, lite::DDim, TARGET(kCUDA)>(x_cpu_data, x_cpu.dims());
  y.Assign<float, lite::DDim, TARGET(kCUDA)>(y_cpu_data, y_cpu.dims());

  param.X = &x;
  param.Y = &y;
  param.Out = &out;
  param.axis = -1;
  elementwise_add_kernel.SetParam(param);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);

  elementwise_add_kernel.SetContext(std::move(ctx));
  elementwise_add_kernel.Launch();
  cudaDeviceSynchronize();

  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * out.numel(), IoDirection::DtoH);
  ElementwiseBroadcastRef(
      x_ref_data, y_ref_data, out_ref_data, n * h * w, c, 1);
  for (int i = 0; i < out.numel(); i++) {
    EXPECT_NEAR(out_cpu_data[i], out_ref_data[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
