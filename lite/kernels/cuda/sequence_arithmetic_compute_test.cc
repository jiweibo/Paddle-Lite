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

#include "lite/kernels/cuda/sequence_arithmetic_compute.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "lite/api/test_helper.h"
#include "lite/core/op_registry.h"
#include "lite/utils/float16.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class SequenceArithmeticTest : public ::testing::Test {
 protected:
  SequenceArithmeticTest() : lod_info({{0, 2, 5, 9}}) {
    in_out_init();
    device_init();
  }

  void in_out_init() {
    prepare_input(&x_ref, lod_info);
    prepare_input(&y_ref, lod_info);
    out_ref.Resize(x_ref.dims());

    x_gpu.Resize(x_ref.dims());
    x_gpu.set_lod(x_ref.lod());
    y_gpu.Resize(y_ref.dims());
    y_gpu.set_lod(y_ref.lod());
    out_gpu.Resize(out_ref.dims());
  }

  void prepare_input(Tensor* x, const LoD& x_lod) {
    x->Resize({static_cast<int64_t>(x_lod[0].back()), 3});
    x->set_lod(x_lod);
    auto x_data = x->mutable_data<float>();
    for (int i = 0; i < x->numel(); i++) {
      x_data[i] = (i - x->numel() / 2) * 1.1;
    }
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
    param.X = &x_gpu;
    param.Y = &y_gpu;
    param.Out = &out_gpu;
  }

  void float_data_init() {
    x_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref.data<float>(),
                                                   x_gpu.dims());
    y_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(y_ref.data<float>(),
                                                   y_gpu.dims());
  }

  void half_data_init() {
    x_half.Resize(x_ref.dims());
    auto x_half_data = x_half.mutable_data<half>();
    for (int64_t i = 0; i < x_half.numel(); i++) {
      x_half_data[i] = half(lite::float16(x_ref.data<float>()[i]));
    }
    x_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu.dims());

    y_half.Resize(y_ref.dims());
    auto y_half_data = y_half.mutable_data<half>();
    for (int64_t i = 0; i < y_half.numel(); i++) {
      y_half_data[i] = half(lite::float16(y_ref.data<float>()[i]));
    }
    y_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(y_half_data, y_gpu.dims());
  }

  void cpu_base(const Tensor& x, const Tensor& y, Tensor* out, int op_type) {
    auto x_data = x.data<float>();
    auto y_data = y.data<float>();
    out->Resize(x.dims());
    out->set_lod(x.lod());
    auto out_data = out->mutable_data<float>();
    auto x_seq_offset = x.lod()[0];
    auto y_seq_offset = y.lod()[0];
    int seq_num = x_seq_offset.size() - 1;
    int inner_size = x.numel() / x.dims()[0];

    for (int i = 0; i < seq_num; i++) {
      int len_x = (x_seq_offset[i + 1] - x_seq_offset[i]) * inner_size;
      int len_y = (y_seq_offset[i + 1] - y_seq_offset[i]) * inner_size;
      auto input_x = x_data + x_seq_offset[i] * inner_size;
      auto input_y = y_data + y_seq_offset[i] * inner_size;
      auto t_out = out_data + x_seq_offset[i] * inner_size;
      int len = std::min(len_x, len_y);
      for (int j = 0; j < len; j++) {
        switch (op_type) {
          case 1:
            t_out[j] = input_x[j] + input_y[j];
            break;
          case 2:
            t_out[j] = input_x[j] - input_y[j];
            break;
          case 3:
            t_out[j] = input_x[j] * input_y[j];
            break;
          default:
            break;
        }
      }
      if (len_x > len) {
        memcpy(t_out + len, input_x + len, sizeof(float) * (len_x - len));
      }
    }
  }

  LoD lod_info;
  std::vector<int64_t> shape_info;
  lite::Tensor x_ref, y_ref, out_ref;
  lite::Tensor x_gpu, y_gpu, out_gpu;
  lite::Tensor x_half, y_half;
  lite::Tensor out_cpu;

  operators::SequenceArithmeticParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(SequenceArithmeticTest, TestFP32) {
  float_data_init();
  SequenceArithmeticCompute<float, PRECISION(kFloat)> seq_kernel;

  seq_kernel.SetContext(std::move(ctx));
  std::vector<int> calc_types({1, 2, 3});
  for (auto calc_type : calc_types) {
    param.op_type = calc_type;
    cpu_base(x_ref, y_ref, &out_ref, calc_type);
    seq_kernel.SetParam(param);

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
    LOG(INFO) << calc_type << ", fp32, warmup: " << FLAGS_warmup
              << ", repeats: " << FLAGS_repeats << ", spend "
              << duration / FLAGS_repeats << " ms in average.";

    out_cpu.Resize(out_gpu.dims());
    CopySync<TARGET(kCUDA)>(out_cpu.mutable_data<float>(),
                            out_gpu.data<float>(),
                            sizeof(float) * out_gpu.numel(),
                            IoDirection::DtoH);

    for (int i = 0; i < out_gpu.numel(); ++i) {
      EXPECT_NEAR(out_cpu.data<float>()[i], out_ref.data<float>()[i], 1e-5);
    }
  }
}

TEST_F(SequenceArithmeticTest, TestFP16) {
  half_data_init();
  SequenceArithmeticCompute<__half, PRECISION(kFP16)> seq_kernel;

  seq_kernel.SetContext(std::move(ctx));
  std::vector<int> calc_types({1, 2, 3});
  for (auto calc_type : calc_types) {
    param.op_type = calc_type;
    cpu_base(x_ref, y_ref, &out_ref, calc_type);
    seq_kernel.SetParam(param);

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
    LOG(INFO) << calc_type << ", fp16, warmup: " << FLAGS_warmup
              << ", repeats: " << FLAGS_repeats << ", spend "
              << duration / FLAGS_repeats << " ms in average.";

    out_cpu.Resize(out_gpu.dims());
    out_cpu.set_lod(out_gpu.lod());
    const half* out_gpu_data = out_gpu.data<half>();
    half* out_cpu_data = out_cpu.mutable_data<half>();
    CopySync<TARGET(kCUDA)>(out_cpu_data,
                            out_gpu_data,
                            sizeof(half) * out_gpu.numel(),
                            IoDirection::DtoH);

    for (int i = 0; i < out_cpu.numel(); ++i) {
      float res = static_cast<float>(lite::float16(out_cpu_data[i]));
      float ref = out_ref.data<float>()[i];
      EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 2e-3);
    }
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
