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

#include "lite/kernels/cuda/sequence_reverse_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/api/test_helper.h"
#include "lite/backends/cuda/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class SequenceReverseTest : public ::testing::Test {
 protected:
  SequenceReverseTest()
      : lod_len(10),
        feature_len(4),
        lod_info({{0, 2, 4}, {0, 3, 5, 6, 10}}),
        shape_info({lod_len, feature_len}) {
    x_gpu.Resize(lite::DDim(shape_info));
    x_ref.Resize(lite::DDim(shape_info));
    x_ref.set_lod(lod_info);
    y_gpu.Resize(lite::DDim(shape_info));
    y_ref.Resize(lite::DDim(shape_info));
    y_ref.set_lod(lod_info);

    auto x_ref_data = x_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < x_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    cpu_base(&x_ref, &y_ref);
    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    param.X = &x_gpu;
    param.Out = &y_gpu;
  }

  void float_data_init() {
    x_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref.data<float>(),
                                                   x_gpu.dims());
    x_gpu.set_lod(x_ref.lod());
  }

  void half_data_init() {
    x_half.Resize(lite::DDim(shape_info));
    auto x_half_data = x_half.mutable_data<half>();
    for (int64_t i = 0; i < x_half.numel(); i++) {
      x_half_data[i] = half(lite::cuda::float16(x_ref.data<float>()[i]));
    }
    x_gpu.Assign<__half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu.dims());
    x_gpu.set_lod(x_ref.lod());
  }

  void cpu_base(const lite::Tensor* x, lite::Tensor* y) {
    const auto* x_data = x->data<float>();
    auto seq_offset = x->lod()[x->lod().size() - 1];
    int width = x->numel() / x->dims()[0];
    auto* y_data = y->mutable_data<float>();
    for (int i = 0; i < static_cast<int>(seq_offset.size()) - 1; ++i) {
      auto start_pos = seq_offset[i];
      auto end_pos = seq_offset[i + 1];
      for (auto pos = start_pos; pos < end_pos; ++pos) {
        auto cur_pos = end_pos - pos - 1 + start_pos;
        std::memcpy(y_data + pos * width,
                    x_data + cur_pos * width,
                    width * sizeof(float));
      }
    }
  }

  int lod_len, feature_len;
  LoD lod_info;
  std::vector<int64_t> shape_info;
  lite::Tensor x_ref, y_ref;
  lite::Tensor x_gpu, y_gpu;
  lite::Tensor x_half;
  lite::Tensor y_cpu;

  operators::SequenceReverseParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(SequenceReverseTest, TestFP32) {
  float_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  SequenceReverseCompute<float, PRECISION(kFloat)> seq_kernel;
  seq_kernel.SetParam(param);
  seq_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    seq_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  seq_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    seq_kernel.Run();
  }
  cudaDeviceSynchronize();
  auto duration = (GetCurrentUS() - start) / 1000.0;
  LOG(INFO) << "fp32, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";
  y_cpu.Resize(y_gpu.dims());
  y_cpu.set_lod(y_gpu.lod());
  CopySync<TARGET(kCUDA)>(y_cpu.mutable_data<float>(),
                          y_gpu.data<float>(),
                          sizeof(float) * y_gpu.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < y_gpu.numel(); ++i) {
    EXPECT_NEAR(y_cpu.data<float>()[i], y_ref.data<float>()[i], 5e-4);
  }
}

TEST_F(SequenceReverseTest, TestFP16) {
  half_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  SequenceReverseCompute<__half, PRECISION(kFP16)> seq_kernel;
  seq_kernel.SetParam(param);
  seq_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    seq_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  seq_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    seq_kernel.Run();
  }
  cudaDeviceSynchronize();
  auto duration = (GetCurrentUS() - start) / 1000.0;
  LOG(INFO) << "fp16, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";

  y_cpu.Resize(y_gpu.dims());
  y_cpu.set_lod(y_gpu.lod());
  const __half* y_gpu_data = y_gpu.data<__half>();
  __half* y_cpu_data = y_cpu.mutable_data<__half>();
  CopySync<TARGET(kCUDA)>(y_cpu_data,
                          y_gpu_data,
                          sizeof(__half) * y_gpu.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < y_cpu.numel(); ++i) {
    float res = static_cast<float>(lite::cuda::float16(y_cpu_data[i]));
    float ref = y_ref.data<float>()[i];
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-3);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
