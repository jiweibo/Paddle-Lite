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

#include "lite/kernels/cuda/concat_compute.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/api/test_helper.h"
#include "lite/backends/cuda/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class ConcatTest : public ::testing::Test {
 protected:
  ConcatTest() : n(4), c1(3), c2(4), c3(5), h(38), w(38) {
    x1_ref.Resize({n, c1, h, w});
    x2_ref.Resize({n, c2, h, w});
    x3_ref.Resize({n, c3, h, w});
    y_ref.Resize({n, c1 + c2 + c3, h, w});
    x1_gpu.Resize(x1_ref.dims());
    x2_gpu.Resize(x2_ref.dims());
    x3_gpu.Resize(x3_ref.dims());
    y_gpu.Resize({n, c1 + c2 + c3, h, w});

    auto x1_ref_data = x1_ref.mutable_data<float>();
    auto x2_ref_data = x2_ref.mutable_data<float>();
    auto x3_ref_data = x3_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < x1_ref.numel(); i++) {
      x1_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < x2_ref.numel(); i++) {
      x2_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < x3_ref.numel(); i++) {
      x3_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    cpu_base(std::vector<lite::Tensor*>({&x1_ref, &x2_ref, &x3_ref}), &y_ref);
    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    param.x = std::vector<lite::Tensor*>({&x1_gpu, &x2_gpu, &x3_gpu});
    param.axis = 1;
    param.output = &y_gpu;
  }

  void float_data_init() {
    x1_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(x1_ref.data<float>(),
                                                    x1_gpu.dims());
    x2_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(x2_ref.data<float>(),
                                                    x2_gpu.dims());
    x3_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(x3_ref.data<float>(),
                                                    x3_gpu.dims());
  }

  void half_data_init() {
    x1_half.Resize(x1_ref.dims());
    x2_half.Resize(x2_ref.dims());
    x3_half.Resize(x3_ref.dims());
    auto x1_half_data = x1_half.mutable_data<half>();
    auto x2_half_data = x2_half.mutable_data<half>();
    auto x3_half_data = x3_half.mutable_data<half>();
    for (int64_t i = 0; i < x1_half.numel(); i++) {
      x1_half_data[i] = half(lite::cuda::float16(x1_ref.data<float>()[i]));
    }
    for (int64_t i = 0; i < x2_half.numel(); i++) {
      x2_half_data[i] = half(lite::cuda::float16(x2_ref.data<float>()[i]));
    }
    for (int64_t i = 0; i < x3_half.numel(); i++) {
      x3_half_data[i] = half(lite::cuda::float16(x3_ref.data<float>()[i]));
    }
    x1_gpu.Assign<__half, lite::DDim, TARGET(kCUDA)>(x1_half_data,
                                                     x1_gpu.dims());
    x2_gpu.Assign<__half, lite::DDim, TARGET(kCUDA)>(x2_half_data,
                                                     x2_gpu.dims());
    x3_gpu.Assign<__half, lite::DDim, TARGET(kCUDA)>(x3_half_data,
                                                     x3_gpu.dims());
  }

  void cpu_base(const std::vector<lite::Tensor*>& input,
                lite::Tensor* output,
                const int axis = 1) {
    int num = input.size();
    int rows = 1;
    auto dim_0 = input[0]->dims();
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int out_rows = rows, out_cols = 0;

    std::vector<int> input_cols(num, 0);
    for (int i = 0; i < num; ++i) {
      int input_i_numel = input[i]->numel();
      int t_cols = input_i_numel / rows;
      out_cols += t_cols;
      input_cols[i] = t_cols;
    }

    auto output_data = output->mutable_data<float>();
    int col_idx = 0;
    for (int j = 0; j < num; ++j) {
      int col_len = input_cols[j];
      auto input_data = input[j]->data<float>();
      for (int k = 0; k < out_rows; ++k) {
        memcpy(output_data + k * out_cols + col_idx,
               input_data + k * col_len,
               sizeof(float) * col_len);
      }
      col_idx += col_len;
    }
  }

  int n, c1, c2, c3, h, w;
  lite::Tensor x1_ref, x2_ref, x3_ref, y_ref;
  lite::Tensor x1_gpu, x2_gpu, x3_gpu, y_gpu;
  lite::Tensor x1_half, x2_half, x3_half;
  lite::Tensor y_cpu;

  operators::ConcatParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(ConcatTest, TestFP32) {
  float_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  ConcatCompute<float, PRECISION(kFloat)> concat_kernel;
  concat_kernel.SetParam(param);
  concat_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    concat_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  concat_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    concat_kernel.Run();
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

TEST_F(ConcatTest, TestFP16) {
  half_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  ConcatCompute<__half, PRECISION(kFP16)> concat_kernel;
  concat_kernel.SetParam(param);
  concat_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    concat_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  concat_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    concat_kernel.Run();
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
