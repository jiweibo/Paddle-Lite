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

#include "lite/kernels/cuda/search_group_padding_compute.h"

#include <gtest/gtest.h>

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

class SearchGroupPaddingTest : public ::testing::Test {
 protected:
  SearchGroupPaddingTest() : x_dims0(2), x_dims1(3), x_lod({{0, 1}}) {
    in_out_init();
    cpu_base(&out_emb_ref, &out_new_ref, &out_padding_ref);
    device_init();
  }

  void in_out_init() {
    x_ref.Resize({x_dims0, x_dims1});
    x_ref.set_lod(x_lod);

    out_emb_ref.Resize({1, x_dims1});
    out_emb_gpu.Resize({1, x_dims1});
    out_new_ref.Resize({x_dims0, 1});
    out_new_gpu.Resize({x_dims0, 1});
    out_padding_ref.Resize({1, 1});
    out_padding_gpu.Resize({1, 1});

    // prepare input
    auto x_ref_data = x_ref.mutable_data<float>();
    for (int64_t i = 0; i < x_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i);
    }
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
    param.x = &x_gpu;
    param.out_emb_padding = &out_emb_gpu;
    param.out_new = &out_new_gpu;
    param.out_padding = &out_padding_gpu;
  }

  void float_data_init() {
    x_gpu.Resize(x_ref.dims());
    x_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref.data<float>(),
                                                   x_gpu.dims());
    x_gpu.set_lod(x_ref.lod());
  }

  void half_data_init() {
    x_half.Resize(x_ref.dims());
    auto x_half_data = x_half.mutable_data<half>();
    for (int64_t i = 0; i < x_half.numel(); i++) {
      x_half_data[i] = half(lite::cuda::float16(x_ref.data<float>()[i]));
    }
    x_gpu.Resize(x_ref.dims());
    x_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu.dims());
    x_gpu.set_lod(x_ref.lod());
  }

  void cpu_base(Tensor* out_emb_ref,
                Tensor* out_new_ref,
                Tensor* out_padding_ref) {
    auto* out_emb_ref_data = out_emb_ref->mutable_data<float>();
    out_emb_ref_data[0] = 0.f;
    out_emb_ref_data[1] = 1.f;
    out_emb_ref_data[2] = 2.f;
    auto* out_new_ref_data = out_new_ref->mutable_data<float>();
    out_new_ref_data[0] = 0.f;
    out_new_ref_data[1] = 0.f;
    auto* out_padding_ref_data = out_padding_ref->mutable_data<float>();
    out_padding_ref_data[0] = 0.f;
  }

  int x_dims0, x_dims1;
  LoD x_lod;
  lite::Tensor x_ref, out_emb_ref, out_new_ref, out_padding_ref;
  lite::Tensor x_gpu, out_emb_gpu, out_new_gpu, out_padding_gpu;
  lite::Tensor x_half;
  lite::Tensor out_emb_cpu, out_new_cpu, out_padding_cpu;

  operators::SearchGroupPaddingParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(SearchGroupPaddingTest, TestFP32) {
  float_data_init();
  SearchGroupPaddingCompute<float, PRECISION(kFloat)> kernel;
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

  out_emb_cpu.Resize(out_emb_gpu.dims());
  out_new_cpu.Resize(out_new_gpu.dims());
  out_padding_cpu.Resize(out_padding_gpu.dims());
  CopySync<TARGET(kCUDA)>(out_emb_cpu.mutable_data<float>(),
                          out_emb_gpu.data<float>(),
                          sizeof(float) * out_emb_gpu.numel(),
                          IoDirection::DtoH);
  CopySync<TARGET(kCUDA)>(out_new_cpu.mutable_data<float>(),
                          out_new_gpu.data<float>(),
                          sizeof(float) * out_new_gpu.numel(),
                          IoDirection::DtoH);
  CopySync<TARGET(kCUDA)>(out_padding_cpu.mutable_data<float>(),
                          out_padding_gpu.data<float>(),
                          sizeof(float) * out_padding_gpu.numel(),
                          IoDirection::DtoH);
  for (int i = 0; i < out_emb_cpu.numel(); ++i) {
    EXPECT_NEAR(
        out_emb_cpu.data<float>()[i], out_emb_ref.data<float>()[i], 1e-5);
  }
  for (int i = 0; i < out_new_cpu.numel(); i++) {
    EXPECT_NEAR(
        out_new_cpu.data<float>()[i], out_new_ref.data<float>()[i], 1e-5);
  }
  for (int i = 0; i < out_padding_cpu.numel(); i++) {
    EXPECT_NEAR(out_padding_cpu.data<float>()[i],
                out_padding_ref.data<float>()[i],
                1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
