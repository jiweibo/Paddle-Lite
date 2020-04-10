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

#include "lite/kernels/cuda/lookup_table_compute.h"
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/api/test_helper.h"
#include "lite/backends/cuda/float16.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

class LookUpTableTest : public ::testing::Test {
 protected:
  LookUpTableTest()
      : vocab_size(512),
        emb_size(15),
        ids_h(3),
        ids_w(1),
        padding_idx(-1),
        ids_shape({ids_h, ids_w}),
        w_shape({vocab_size, emb_size}),
        out_shape({ids_h, ids_w, emb_size}) {
    Ids_gpu.Resize(lite::DDim(ids_shape));
    Ids_ref.Resize(lite::DDim(ids_shape));

    W_gpu.Resize(lite::DDim(w_shape));
    W_ref.Resize(lite::DDim(w_shape));

    auto ids_ref_data = Ids_ref.mutable_data<int64_t>();
    auto w_ref_data = W_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < Ids_ref.dims().production(); i++) {
      ids_ref_data[i] = i % vocab_size;
    }
    for (int64_t i = 0; i < W_ref.dims().production(); i++) {
      w_ref_data[i] =
          static_cast<float>(i + 1) / (W_ref.dims().production() + 1);
    }

    Out_ref.Resize(lite::DDim(out_shape));
    Out_cpu.Resize(lite::DDim(out_shape));
    Out_gpu.Resize(lite::DDim(out_shape));
    fc_cpu_base(&Ids_ref, &W_ref, &Out_ref, padding_idx);

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    param.Ids = &Ids_gpu;
    param.W = &W_gpu;
    param.Out = &Out_gpu;
    param.padding_idx = padding_idx;
  }

  void float_data_init() {
    Ids_gpu.Assign<int64_t, lite::DDim, TARGET(kCUDA)>(Ids_ref.data<int64_t>(),
                                                       Ids_gpu.dims());
    W_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(W_ref.data<float>(),
                                                   W_gpu.dims());
  }

  void half_data_init() {
    W_half.Resize(lite::DDim(w_shape));
    auto w_half_data = W_half.mutable_data<__half>();
    for (int64_t i = 0; i < W_half.dims().production(); i++) {
      w_half_data[i] = half(lite::cuda::float16(W_ref.data<float>()[i]));
    }
    Ids_gpu.Assign<int64_t, lite::DDim, TARGET(kCUDA)>(Ids_ref.data<int64_t>(),
                                                       Ids_gpu.dims());
    W_gpu.Assign<__half, lite::DDim, TARGET(kCUDA)>(w_half_data, W_gpu.dims());
  }

  void fc_cpu_base(const lite::Tensor* ids_t,
                   const lite::Tensor* table_t,
                   lite::Tensor* output_t,
                   int64_t padding_idx) {
    auto* ids = ids_t->data<int64_t>();
    int64_t ids_numel = ids_t->dims().production();

    int64_t row_number = table_t->dims()[0];
    int64_t row_width = table_t->dims()[1];

    auto* table = table_t->data<float>();
    auto* output = output_t->mutable_data<float>();
    memset(output, 0, output_t->dims().production() * sizeof(float));
    for (int64_t i = 0; i < ids_numel; ++i) {
      if (padding_idx != -1 && ids[i] == padding_idx) {
        memset(output + i * row_width, 0, row_width * sizeof(float));
      } else {
        CHECK_LT(ids[i], row_number);
        CHECK_GE(ids[i], 0);
        memcpy(output + i * row_width,
               table + ids[i] * row_width,
               row_width * sizeof(float));
      }
    }
  }

  int vocab_size;
  int emb_size;
  int ids_h;
  int ids_w;
  int padding_idx;

  std::vector<int64_t> ids_shape, w_shape, out_shape;
  lite::Tensor Ids_ref, W_ref, Out_ref;
  lite::Tensor Ids_gpu, W_gpu, W_half;
  lite::Tensor Out_cpu, Out_gpu;

  operators::LookupTableParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(LookUpTableTest, TestFP32) {
  float_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  LookupTableCompute<float, PRECISION(kFloat)> lkt_kernel;
  lkt_kernel.SetParam(param);
  lkt_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    lkt_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    lkt_kernel.Launch();
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

  for (int i = 0; i < Out_ref.numel(); ++i) {
    EXPECT_NEAR(Out_cpu.data<float>()[i], Out_ref.data<float>()[i], 1e-5);
  }
}

TEST_F(LookUpTableTest, TestFP16) {
  half_data_init();
  auto& context = ctx->As<CUDAContext>();
  context.SetExecStream(stream);
  LookupTableCompute<__half, PRECISION(kFP16)> lkt_kernel;
  lkt_kernel.SetParam(param);
  lkt_kernel.SetContext(std::move(ctx));

  for (int i = 0; i < FLAGS_warmup; ++i) {
    lkt_kernel.Launch();
    cudaDeviceSynchronize();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    lkt_kernel.Launch();
  }
  cudaDeviceSynchronize();
  auto duration = (GetCurrentUS() - start) / 1000.0;
  LOG(INFO) << "fp16, warmup: " << FLAGS_warmup
            << ", repeats: " << FLAGS_repeats << ", spend "
            << duration / FLAGS_repeats << " ms in average.";

  const __half* out_gpu_data = Out_gpu.data<__half>();
  __half* out_cpu_data = Out_cpu.mutable_data<__half>();
  CopySync<TARGET(kCUDA)>(out_cpu_data,
                          out_gpu_data,
                          sizeof(__half) * Out_gpu.numel(),
                          IoDirection::DtoH);

  for (int i = 0; i < Out_cpu.numel(); ++i) {
    float res = static_cast<float>(lite::cuda::float16(out_cpu_data[i]));
    float ref = Out_ref.data<float>()[i];
    LOG(INFO) << i << ":, ref is " << ref << ", while res is " << res
              << ", diff is " << ref - res;
    EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 1e-3);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
