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

#include "lite/kernels/cuda/sequence_pool_compute.h"

#include <gtest/gtest.h>

#include <cstring>
#include <map>
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

class SequencePoolTest : public ::testing::Test {
 protected:
  SequencePoolTest() : n(10), c(8) {
    lite::LoD lod;
    lod.push_back(std::vector<uint64_t>{0, static_cast<uint64_t>(n)});
    x_ref.Resize({n, c});
    x_ref.set_lod(lod);
    out_ref.Resize({static_cast<int64_t>(lod[0].size()) - 1, c});

    x_gpu.Resize(x_ref.dims());
    x_gpu.set_lod(x_ref.lod());
    out_gpu.Resize({static_cast<int64_t>(lod[0].size()) - 1, c});

    auto x_ref_data = x_ref.mutable_data<float>();

    // prepare input
    for (int64_t i = 0; i < x_ref.numel(); i++) {
      x_ref_data[i] = static_cast<float>(1.1f * i);
    }

    device_init();
  }

  void device_init() {
    ctx.reset(new KernelContext);
    cudaStreamCreate(&stream);
    auto& context = ctx->As<CUDAContext>();
    context.SetExecStream(stream);
  }

  void param_init(const std::string& pool_type) {
    param.X = &x_gpu;
    param.pool_type = pool_type;
    param.Out = &out_gpu;
  }

  void float_data_init() {
    x_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(x_ref.data<float>(),
                                                   x_gpu.dims());
  }

  void half_data_init() {
    x_half.Resize(x_ref.dims());
    auto x_half_data = x_half.mutable_data<half>();
    for (int64_t i = 0; i < x_half.numel(); i++) {
      x_half_data[i] = half(lite::cuda::float16(x_ref.data<float>()[i]));
    }
    x_gpu.Assign<__half, lite::DDim, TARGET(kCUDA)>(x_half_data, x_gpu.dims());
  }

  void cpu_base(lite::Tensor* output, const std::string pool_type = "MAX") {
    std::map<std::string, std::vector<float>> type_map;
    type_map["MAX"] = {79.2, 80.3, 81.4, 82.5, 83.6, 84.7, 85.8, 86.9};
    type_map["AVERAGE"] = {39.6, 40.7, 41.8, 42.9, 44, 45.1, 46.2, 47.3};
    type_map["SUM"] = {396, 407, 418, 429, 440, 451, 462, 473};
    type_map["SQRT"] = {
        125.226, 128.705, 132.183, 135.662, 139.14, 142.619, 146.097, 149.576};
    type_map["FIRST"] = {0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7};
    type_map["LAST"] = {79.2, 80.3, 81.4, 82.5, 83.6, 84.7, 85.8, 86.9};
    auto* out_data = output->mutable_data<float>();
    if (pool_type == "MAX") {
      std::memcpy(out_data,
                  type_map["MAX"].data(),
                  type_map["MAX"].size() * sizeof(float));
    } else if (pool_type == "AVERAGE") {
      std::memcpy(out_data,
                  type_map["AVERAGE"].data(),
                  type_map["AVERAGE"].size() * sizeof(float));
    } else if (pool_type == "SUM") {
      std::memcpy(out_data,
                  type_map["SUM"].data(),
                  type_map["SUM"].size() * sizeof(float));
    } else if (pool_type == "SQRT") {
      std::memcpy(out_data,
                  type_map["SQRT"].data(),
                  type_map["SQRT"].size() * sizeof(float));
    } else if (pool_type == "FIRST") {
      std::memcpy(out_data,
                  type_map["FIRST"].data(),
                  type_map["FIRST"].size() * sizeof(float));
    } else if (pool_type == "LAST") {
      std::memcpy(out_data,
                  type_map["LAST"].data(),
                  type_map["LAST"].size() * sizeof(float));
    } else {
      LOG(FATAL) << "not supported pool type: " << pool_type;
    }
  }

  int n, c;
  lite::Tensor x_ref, out_ref;
  lite::Tensor x_gpu, out_gpu;
  lite::Tensor x_half;
  lite::Tensor out_cpu;

  operators::SequencePoolParam param;
  std::unique_ptr<KernelContext> ctx;
  cudaStream_t stream;
};

TEST_F(SequencePoolTest, TestFP32) {
  float_data_init();
  SequencePoolCompute<float, PRECISION(kFloat)> seq_kernel;

  seq_kernel.SetContext(std::move(ctx));
  std::vector<std::string> pool_types(
      {"MAX", "AVERAGE", "SUM", "SQRT", "FIRST", "LAST"});
  for (std::string pool_type : pool_types) {
    param_init(pool_type);
    cpu_base(&out_ref, pool_type);
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
    LOG(INFO) << pool_type << ", fp32, warmup: " << FLAGS_warmup
              << ", repeats: " << FLAGS_repeats << ", spend "
              << duration / FLAGS_repeats << " ms in average.";

    out_cpu.Resize(out_gpu.dims());
    CopySync<TARGET(kCUDA)>(out_cpu.mutable_data<float>(),
                            out_gpu.data<float>(),
                            sizeof(float) * out_gpu.numel(),
                            IoDirection::DtoH);

    for (int i = 0; i < out_gpu.numel(); ++i) {
      EXPECT_NEAR(out_cpu.data<float>()[i], out_ref.data<float>()[i], 5e-4);
    }
  }
}

TEST_F(SequencePoolTest, TestFP16) {
  half_data_init();
  SequencePoolCompute<__half, PRECISION(kFP16)> seq_kernel;

  seq_kernel.SetContext(std::move(ctx));
  std::vector<std::string> pool_types(
      {"MAX", "AVERAGE", "SUM", "SQRT", "FIRST", "LAST"});
  for (std::string pool_type : pool_types) {
    param_init(pool_type);
    cpu_base(&out_ref, pool_type);
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
    LOG(INFO) << pool_type << ", fp16, warmup: " << FLAGS_warmup
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
      float res = static_cast<float>(lite::cuda::float16(out_cpu_data[i]));
      float ref = out_ref.data<float>()[i];
      EXPECT_NEAR(fabs(res - ref) / (ref + 1e-5), 0., 2e-3);
    }
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
