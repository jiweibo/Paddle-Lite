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

#include "lite/kernels/cuda/search_grnn_compute.h"
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

class SearchGrnnTest : public ::testing::Test {
 protected:
  SearchGrnnTest()
      : num_batch(3),
        num_input(6),
        num_hidden(6),
        x_shape({num_batch, num_input}),
        wi_shape({3, num_hidden, num_input}),
        wh_shape({3, num_hidden, num_hidden}),
        out_shape({num_batch, num_hidden}),
        x_lod({{0, 1, 3}}) {
    X_gpu.Resize(lite::DDim(x_shape));
    X_ref.Resize(lite::DDim(x_shape));
    X_ref.set_lod(x_lod);

    Wi_gpu.Resize(lite::DDim(wi_shape));
    Wi_ref.Resize(lite::DDim(wi_shape));

    Wh_gpu.Resize(lite::DDim(wh_shape));
    Wh_ref.Resize(lite::DDim(wh_shape));

    auto x_ref_data = X_ref.mutable_data<float>();
    auto wi_ref_data = Wi_ref.mutable_data<float>();
    auto wh_ref_data = Wh_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < X_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < Wi_ref.numel(); i++) {
      wi_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }
    for (int64_t i = 0; i < Wh_ref.numel(); i++) {
      wh_ref_data[i] = static_cast<float>(i % 10 * 0.2);
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_cpu.Resize(lite::DDim(out_shape));
    // cpu_base(&X_ref, &W_ref, &b_ref, &Out_ref);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    param.x = &X_gpu;
    param.wi = &Wi_gpu;
    param.wh = &Wh_gpu;
    param.out = &Out_gpu;
    param.idx_sorted_by_width = &idx_sorted_by_width;
    param.layout_input = &layout_input;
    param.tmp_buffer = &tmp_buffer;
    param.num_input = num_input;
    param.num_hidden = num_hidden;
  }

  void float_data_init() {
    X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(X_ref.data<float>(),
                                                   X_gpu.dims());
    X_gpu.set_lod(X_ref.lod());
    Wi_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(Wi_ref.data<float>(),
                                                    Wi_gpu.dims());
    Wh_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(Wh_ref.data<float>(),
                                                    Wh_gpu.dims());
  }

  void half_data_init() {
    X_half.Resize(lite::DDim(x_shape));
    auto x_half_data = X_half.mutable_data<half>();
    for (int64_t i = 0; i < X_half.numel(); i++) {
      x_half_data[i] = half(lite::float16(X_ref.data<float>()[i]));
    }
    X_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(x_half_data, X_gpu.dims());
    X_gpu.set_lod(X_ref.lod());

    Wi_half.Resize(Wi_ref.dims());
    auto wi_half_data = Wi_half.mutable_data<half>();
    for (int64_t i = 0; i < Wi_half.numel(); i++) {
      wi_half_data[i] = half(lite::float16(Wi_ref.data<float>()[i]));
    }
    Wi_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(wi_half_data, Wi_gpu.dims());

    Wh_half.Resize(Wh_ref.dims());
    auto wh_half_data = Wh_half.mutable_data<half>();
    for (int64_t i = 0; i < Wh_half.numel(); i++) {
      wh_half_data[i] = half(lite::float16(Wh_ref.data<float>()[i]));
    }
    Wh_gpu.Assign<half, lite::DDim, TARGET(kCUDA)>(wh_half_data, Wh_gpu.dims());
  }

  void cpu_base(const lite::Tensor* X,
                const lite::Tensor* W,
                const lite::Tensor* b,
                lite::Tensor* Out) {}

  int num_batch, num_input, num_hidden;
  std::vector<int64_t> x_shape, wi_shape, wh_shape, out_shape;
  LoD x_lod;
  lite::Tensor X_ref, Wi_ref, Wh_ref, Out_ref;
  lite::Tensor X_gpu, Wi_gpu, Wh_gpu;
  lite::Tensor X_half, Wi_half, Wh_half;
  lite::Tensor idx_sorted_by_width, layout_input, tmp_buffer;
  lite::Tensor Out_cpu, Out_gpu;

  operators::SearchGrnnParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(SearchGrnnTest, TestFP32) {
  float_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  SearchGrnnCompute<float, PRECISION(kFloat)> search_grnn_kernel;
  search_grnn_kernel.SetParam(param);
  search_grnn_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    search_grnn_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  search_grnn_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    search_grnn_kernel.Run();
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
    LOG(INFO) << Out_cpu.data<float>()[i];
  }
}

TEST_F(SearchGrnnTest, TestFP16) {
  half_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  SearchGrnnCompute<half, PRECISION(kFP16)> search_grnn_kernel;
  search_grnn_kernel.SetParam(param);
  search_grnn_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    search_grnn_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  search_grnn_kernel.PrepareForRun();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    search_grnn_kernel.Run();
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
    LOG(INFO) << res;
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
