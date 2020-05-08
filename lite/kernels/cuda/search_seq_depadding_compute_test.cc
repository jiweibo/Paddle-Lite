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

#include "lite/kernels/cuda/search_seq_depadding_compute.h"
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

using Tensor = lite::Tensor;

class SearchSeqDepaddingTest : public ::testing::Test {
 protected:
  SearchSeqDepaddingTest()
      : pad_dim0(6),
        pad_dim1(4),
        src_dim(3),
        pad_lod({{0, 4, 6}}),
        src_lod({{0, 2, 3}}) {
    in_out_init();
    cpu_base(&out_ref);
    device_init();
  }

  void in_out_init() {
    pad_ref.Resize({pad_dim0, pad_dim1});
    src_ref.Resize({src_dim, 1});
    out_ref.Resize({src_dim, pad_dim1});
    pad_ref.set_lod(pad_lod);
    src_ref.set_lod(src_lod);

    pad_gpu.Resize(pad_ref.dims());
    src_gpu.Resize(src_ref.dims());
    out_gpu.Resize(out_ref.dims());
    pad_gpu.set_lod(pad_ref.lod());
    src_gpu.set_lod(src_ref.lod());

    auto* pad_cpu_data = pad_ref.mutable_data<float>();
    auto* src_cpu_data = src_ref.mutable_data<float>();
    for (int i = 0; i < pad_ref.numel(); ++i) {
      pad_cpu_data[i] = static_cast<float>(i);
    }
    for (int i = 0; i < src_ref.numel(); ++i) {
      src_cpu_data[i] = static_cast<float>(i);
    }
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
    param.pad = &pad_gpu;
    param.src = &src_gpu;
    param.out = &out_gpu;
  }

  void float_data_init() {
    pad_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(pad_ref.data<float>(),
                                                     pad_gpu.dims());
    src_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(src_ref.data<float>(),
                                                     src_gpu.dims());
  }

  void half_data_init() {
    pad_half.Resize(pad_ref.dims());
    auto pad_half_data = pad_half.mutable_data<half>();
    for (int64_t i = 0; i < pad_half.numel(); i++) {
      pad_half_data[i] = half(lite::cuda::float16(pad_ref.data<float>()[i]));
    }
    pad_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(pad_half_data,
                                                    pad_gpu.dims());

    src_half.Resize(src_ref.dims());
    auto src_half_data = src_half.mutable_data<half>();
    for (int64_t i = 0; i < src_half.numel(); i++) {
      src_half_data[i] = half(lite::cuda::float16(src_ref.data<float>()[i]));
    }
    src_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(src_half_data,
                                                    src_gpu.dims());
  }

  void cpu_base(lite::Tensor* out_ref) {
    std::vector<float> ref_results = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19};
    std::memcpy(out_ref->mutable_data<float>(),
                ref_results.data(),
                ref_results.size() * sizeof(float));
  }

  int pad_dim0, pad_dim1, src_dim;
  LoD pad_lod, src_lod;
  lite::Tensor pad_ref, src_ref, out_ref;
  lite::Tensor pad_gpu, src_gpu, out_gpu;
  lite::Tensor pad_half, src_half;
  lite::Tensor out_cpu;

  operators::SearchSeqDepaddingParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(SearchSeqDepaddingTest, TestFP32) {
  float_data_init();
  SearchSeqDepaddingCompute<float, PRECISION(kFloat)> seq_kernel;
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
  out_cpu.Resize(out_gpu.dims());
  out_cpu.set_lod(out_gpu.lod());
  CopySync<TARGET(kCUDA)>(out_cpu.mutable_data<float>(),
                          out_gpu.data<float>(),
                          sizeof(float) * out_gpu.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < out_gpu.numel(); ++i) {
    EXPECT_NEAR(out_cpu.data<float>()[i], out_ref.data<float>()[i], 1e-5);
  }
}

TEST_F(SearchSeqDepaddingTest, TestFP16) {
  half_data_init();
  SearchSeqDepaddingCompute<__half, PRECISION(kFP16)> seq_kernel;
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

  out_cpu.Resize(out_gpu.dims());
  out_cpu.set_lod(out_gpu.lod());
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
