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

#include "device_kernels.cuh"
#include "device_utils.cuh"
#include "test_array.h"

namespace {

constexpr size_t max_threads_per_block = 1024;

template<nixl_gpu_level_t level>
__device__ size_t
threadsPerRequest() {
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        return 1;
    } else if constexpr (level == nixl_gpu_level_t::WARP) {
        return warpSize;
    } else {
        return max_threads_per_block;
    }
}

template<nixl_gpu_level_t level>
__device__ size_t
getStatusIndex() {
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        return threadIdx.x;
    } else if constexpr (level == nixl_gpu_level_t::WARP) {
        return threadIdx.x / warpSize;
    } else {
        return 0;
    }
}

template<nixl_gpu_level_t level>
size_t
sharedRequestCount(size_t num_threads) {
    if constexpr (level == nixl_gpu_level_t::THREAD) {
        return num_threads;
    } else if constexpr (level == nixl_gpu_level_t::WARP) {
        // Use 32 for calculation since warpSize is not available on host
        return (num_threads + 31) / 32;
    } else {
        return 1;
    }
}

template<nixl_gpu_level_t level>
__device__ nixl_status_t
doOperation(const nixlDeviceKernelParams &params, nixlGpuXferStatusH *req_ptr) {
    nixl_status_t status;

    switch (params.operation) {
    case nixl_device_operation_t::SINGLE_WRITE:
        status = nixlGpuPostSingleWriteXferReq<level>(params.reqHandle,
                                                      params.singleWrite.index,
                                                      params.singleWrite.localOffset,
                                                      params.singleWrite.remoteOffset,
                                                      params.singleWrite.size,
                                                      params.singleWrite.channelId,
                                                      params.withNoDelay,
                                                      req_ptr);
        break;

    case nixl_device_operation_t::PARTIAL_WRITE:
        status = nixlGpuPostPartialWriteXferReq<level>(params.reqHandle,
                                                       params.partialWrite.count,
                                                       params.partialWrite.descIndices,
                                                       params.partialWrite.sizes,
                                                       params.partialWrite.localOffsets,
                                                       params.partialWrite.remoteOffsets,
                                                       params.partialWrite.signalDescIndex,
                                                       params.partialWrite.signalInc,
                                                       params.partialWrite.signalOffset,
                                                       params.partialWrite.channelId,
                                                       params.withNoDelay,
                                                       req_ptr);
        break;

    case nixl_device_operation_t::WRITE:
        status = nixlGpuPostWriteXferReq<level>(params.reqHandle,
                                                params.write.signalInc,
                                                params.write.channelId,
                                                params.withNoDelay,
                                                req_ptr);
        break;

    case nixl_device_operation_t::SIGNAL_POST:
        status = nixlGpuPostSignalXferReq<level>(params.reqHandle,
                                                 params.signalPost.signalDescIndex,
                                                 params.signalPost.signalInc,
                                                 params.signalPost.signalOffset,
                                                 params.signalPost.channelId,
                                                 params.withNoDelay,
                                                 req_ptr);
        break;

    case nixl_device_operation_t::SIGNAL_WAIT: {
        if (params.signalWait.signalAddr == nullptr) {
            status = NIXL_ERR_INVALID_PARAM;
            break;
        }

        uint64_t value;
        do {
            value = nixlGpuReadSignal<level>(params.signalWait.signalAddr);
        } while (value != params.signalWait.expectedValue);

        status = NIXL_SUCCESS;
        break;
    }

    case nixl_device_operation_t::SIGNAL_WRITE:
        if (params.signalWrite.signalAddr == nullptr) {
            status = NIXL_ERR_INVALID_PARAM;
            break;
        }
        nixlGpuWriteSignal<level>(params.signalWrite.signalAddr, params.signalWrite.value);
        status = NIXL_SUCCESS;
        break;

    default:
        status = NIXL_ERR_INVALID_PARAM;
        break;
    }

    if (status != NIXL_IN_PROG) {
        return (status == NIXL_SUCCESS) ? NIXL_SUCCESS : NIXL_ERR_BACKEND;
    }

    if (!params.withNoDelay || (req_ptr == nullptr)) {
        return NIXL_SUCCESS;
    }

    do {
        status = nixlGpuGetXferStatus<level>(*req_ptr);
    } while (status == NIXL_IN_PROG);

    return status;
}

template<nixl_gpu_level_t level>
__device__ void
kernelJob(const nixlDeviceKernelParams &params,
          nixlDeviceKernelResult *result_ptr,
          nixlGpuXferStatusH *shared_reqs) {
    if (result_ptr == nullptr) {
        return;
    }

    nixl_status_t &status = result_ptr->status;

    if (blockDim.x > max_threads_per_block) {
        status = NIXL_ERR_INVALID_PARAM;
        return;
    }

    if (params.numIters == 0) {
        status = NIXL_ERR_INVALID_PARAM;
        return;
    }

    nixlGpuXferStatusH *req_ptr = nullptr;
    if (params.withRequest) {
        const size_t req_index = getStatusIndex<level>();
        req_ptr = &shared_reqs[req_index];
    }

    for (size_t i = 0; i < params.numIters - 1; i++) {
        status = doOperation<level>(params, req_ptr);
        if (status != NIXL_SUCCESS) {
            return;
        }
    }

    // Last iteration forces completion to ensure all operations are finished
    nixlDeviceKernelParams params_force_completion = params;
    params_force_completion.withNoDelay = true;
    nixlGpuXferStatusH *status_ptr = nullptr;
    if (params.withRequest) {
        const size_t req_index = getStatusIndex<level>();
        status_ptr = &shared_reqs[req_index];
    }
    status = doOperation<level>(params_force_completion, status_ptr);
}

template<nixl_gpu_level_t level>
__global__ void
nixlTestKernel(const nixlDeviceKernelParams params, nixlDeviceKernelResult *result_ptr) {
    extern __shared__ nixlGpuXferStatusH shared_reqs[];
    kernelJob<level>(params, result_ptr, shared_reqs);
    __threadfence_system();
}

} // namespace

nixlDeviceKernelResult
launchNixlDeviceKernel(const nixlDeviceKernelParams &params) {
    testArray<nixlDeviceKernelResult> result(1);
    nixlDeviceKernelResult init_result{NIXL_ERR_INVALID_PARAM};
    result.copyFromHost(&init_result, 1);

    size_t shared_mem_size = 0;
    switch (params.level) {
    case nixl_gpu_level_t::THREAD:
        shared_mem_size = sharedRequestCount<nixl_gpu_level_t::THREAD>(params.numThreads) *
            sizeof(nixlGpuXferStatusH);
        nixlTestKernel<nixl_gpu_level_t::THREAD>
            <<<params.numBlocks, params.numThreads, shared_mem_size>>>(params, result.get());
        break;
    case nixl_gpu_level_t::WARP:
        shared_mem_size = sharedRequestCount<nixl_gpu_level_t::WARP>(params.numThreads) *
            sizeof(nixlGpuXferStatusH);
        nixlTestKernel<nixl_gpu_level_t::WARP>
            <<<params.numBlocks, params.numThreads, shared_mem_size>>>(params, result.get());
        break;
    case nixl_gpu_level_t::BLOCK:
        shared_mem_size = sharedRequestCount<nixl_gpu_level_t::BLOCK>(params.numThreads) *
            sizeof(nixlGpuXferStatusH);
        nixlTestKernel<nixl_gpu_level_t::BLOCK>
            <<<params.numBlocks, params.numThreads, shared_mem_size>>>(params, result.get());
        break;
    default: {
        nixlDeviceKernelResult error_result{NIXL_ERR_INVALID_PARAM};
        return error_result;
    }
    }

    const nixl_status_t sync_status = checkCudaErrors();
    if (sync_status != NIXL_SUCCESS) {
        nixlDeviceKernelResult error_result{sync_status};
        return error_result;
    }

    nixlDeviceKernelResult host_result;
    result.copyToHost(&host_result, 1);
    return host_result;
}
