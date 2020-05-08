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

#include "lite/kernels/cuda/attention_padding_mask_compute.h"

#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "lite/api/test_helper.h"
#include "lite/backends/cuda/float16.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class AttentionPaddingMaskTest : public ::testing::Test {
 protected:
  AttentionPaddingMaskTest()
      : dims(54), lod_info({{0, 54}}), mask(-90000000.f), pad_id(12800001) {
    in_out_init();
    cpu_base(x_ref, y_ref, &out_ref);
    device_init();
  }

  void in_out_init() {
    x_ref.Resize({dims, dims});
    x_ref.set_lod(lod_info);
    y_ref.Resize({dims, 1});
    y_ref.set_lod(lod_info);
    out_ref.Resize(x_ref.dims());
    pad_begin_ref.Resize({static_cast<int64_t>(y_ref.lod()[0].size() - 1)});

    x_gpu.Resize(x_ref.dims());
    y_gpu.Resize(y_ref.dims());
    x_gpu.set_lod(x_ref.lod());
    y_gpu.set_lod(y_ref.lod());
    out_gpu.Resize(out_ref.dims());
    pad_begin_gpu.Resize(pad_begin_ref.dims());

    // prepare input
    auto x_ref_data = x_ref.mutable_data<float>();
    for (int64_t i = 0; i < x_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    auto y_ref_data = y_ref.mutable_data<float>();
    for (int64_t i = 0; i < y_ref.numel(); i++) {
      y_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);

    param.X = &x_gpu;
    param.Y = &y_gpu;
    param.pad_id = pad_id;
    param.mask = mask;
    param.Out = &out_gpu;
    param.pad_begin = &pad_begin_gpu;
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
      x_half_data[i] = half(lite::cuda::float16(x_ref.data<float>()[i]));
    }
    x_gpu.Assign<__half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu.dims());

    y_half.Resize(y_ref.dims());
    auto y_half_data = y_half.mutable_data<half>();
    for (int64_t i = 0; i < y_half.numel(); i++) {
      y_half_data[i] = half(lite::cuda::float16(y_ref.data<float>()[i]));
    }
    y_gpu.Assign<__half, lite::DDim, TARGET(kCUDA)>(y_half_data, y_gpu.dims());
  }

  void cpu_base(const lite::Tensor& x, const lite::Tensor& y, Tensor* out) {
    auto attn_offset = x.lod()[0];
    auto src_offset = y.lod()[0];
    int attn_seq_num = attn_offset.size() - 1;
    int src_seq_num = src_offset.size() - 1;
    int attn_seq_len = attn_offset[1];
    int src_seq_len = x.dims()[1];
    CHECK_EQ(attn_seq_num % src_seq_num, 0);

    auto count = x.numel();
    auto attn_data = x.data<float>();
    out->Resize(x.dims());
    out->set_lod(x.lod());
    auto out_data = out->mutable_data<float>();
    memcpy(out_data, attn_data, count * sizeof(float));

    for (int i = 0; i < attn_seq_num; ++i) {
      for (int j = 0; j < attn_seq_len; ++j) {
        auto tmp_out_data = out_data + src_seq_len * (attn_seq_len * i + j);
        int src_seq_idx = i % src_seq_num;
        int cur_len = src_offset[src_seq_idx + 1] - src_offset[src_seq_idx];
        for (int k = cur_len; k < src_seq_len; k++) {
          tmp_out_data[k] = mask;
        }
      }
    }
  }

  int dims;
  LoD lod_info;
  float mask;
  int pad_id;
  lite::Tensor x_ref, y_ref, out_ref, pad_begin_ref;
  lite::Tensor x_gpu, y_gpu, out_gpu, pad_begin_gpu;
  lite::Tensor x_half, y_half;
  lite::Tensor out_cpu, pad_begin_cpu;

  operators::AttentionPaddingMaskParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(AttentionPaddingMaskTest, TestFP32) {
  float_data_init();
  AttentionPaddingMaskCompute<float, PRECISION(kFloat)> kernel;
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
  out_cpu.Resize(out_gpu.dims());
  CopySync<TARGET(kCUDA)>(out_cpu.mutable_data<float>(),
                          out_gpu.data<float>(),
                          sizeof(float) * out_gpu.numel(),
                          IoDirection::DtoH);
  pad_begin_cpu.Resize(pad_begin_gpu.dims());
  CopySync<TARGET(kCUDA)>(pad_begin_cpu.mutable_data<int>(),
                          pad_begin_gpu.data<int>(),
                          sizeof(int) * pad_begin_gpu.numel(),
                          IoDirection::DtoH);
  for (int i = 0; i < out_gpu.numel(); ++i) {
    EXPECT_NEAR(out_cpu.data<float>()[i], out_ref.data<float>()[i], 1e-5);
  }
}

TEST_F(AttentionPaddingMaskTest, TestFP16) {
  half_data_init();
  AttentionPaddingMaskCompute<__half, PRECISION(kFP16)> kernel;
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

  out_cpu.Resize(out_gpu.dims());
  const __half* out_gpu_data = out_gpu.data<__half>();
  __half* out_cpu_data = out_cpu.mutable_data<__half>();
  CopySync<TARGET(kCUDA)>(out_cpu_data,
                          out_gpu_data,
                          sizeof(__half) * out_gpu.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < out_cpu.numel(); ++i) {
    float res = static_cast<float>(lite::cuda::float16(out_cpu_data[i]));
    float ref = out_ref.data<float>()[i];
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-3);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
