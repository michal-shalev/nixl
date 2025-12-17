/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common/device_test_base.cuh"

namespace gtest::nixl::gpu::write {

namespace {

class WriteTest : public DeviceApiTestBase<DeviceTestParams> {
protected:
    void runWrite(const TestSetupData &setup_data,
                  size_t num_iters,
                  uint64_t signal_inc) {
        constexpr unsigned channel_id = defaultChannelId;

        NixlDeviceKernelParams params = {};
        params.operation = NixlDeviceOperation::WRITE;
        params.level = getLevel();
        params.numThreads = defaultNumThreads;
        params.numBlocks = 1;
        params.numIters = num_iters;
        params.reqHandle = setup_data.gpuReqHandle;

        applySendMode(params, getSendMode());

        params.write.signalInc = signal_inc;
        params.write.channelId = channel_id;

        const auto result = launchNixlDeviceKernel(params);
        ASSERT_EQ(result.status, NIXL_SUCCESS)
            << "Kernel execution failed with status: " << result.status;
    }
};

} // namespace

TEST_P(WriteTest, Basic) {
    const std::vector<size_t> sizes(defaultBufferCount, defaultBufferSize);

    TestSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWithSignal(sizes, VRAM_SEG, setup_data));

    ASSERT_NO_FATAL_FAILURE(initializeTestData(sizes, setup_data));
    ASSERT_NO_FATAL_FAILURE(runWrite(setup_data, defaultNumIters,
                                     testSignalIncrement));
    ASSERT_NO_FATAL_FAILURE(verifyTestData(sizes, setup_data));
}

TEST_P(WriteTest, WithoutSignal) {
    const std::vector<size_t> sizes(defaultBufferCount, defaultBufferSize);
    constexpr uint64_t signal_inc = 0;

    TestSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWithSignal(sizes, VRAM_SEG, setup_data));

    ASSERT_NO_FATAL_FAILURE(initializeTestData(sizes, setup_data));
    ASSERT_NO_FATAL_FAILURE(runWrite(setup_data, 1000, signal_inc));
    ASSERT_NO_FATAL_FAILURE(verifyTestData(sizes, setup_data));
}

TEST_P(WriteTest, SignalOnly) {
    const std::vector<size_t> sizes;

    TestSetupData setup_data;
    auto guard = setup_data.makeCleanupGuard(this);
    ASSERT_NO_FATAL_FAILURE(setupWithSignal(sizes, VRAM_SEG, setup_data));

    ASSERT_NO_FATAL_FAILURE(runWrite(setup_data, 1000,
                                     testSignalIncrement));
}

} // namespace gtest::nixl::gpu::write

using gtest::nixl::gpu::write::WriteTest;

INSTANTIATE_TEST_SUITE_P(ucxDeviceApi,
                         WriteTest,
                         testing::ValuesIn(WriteTest::getPartialWriteDeviceTestParams()),
                         TestNameGenerator::device);
