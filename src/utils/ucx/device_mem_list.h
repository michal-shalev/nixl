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

#ifndef NIXL_SRC_UTILS_UCX_DEVICE_MEM_LIST_H
#define NIXL_SRC_UTILS_UCX_DEVICE_MEM_LIST_H

#include <memory>
#include <vector>

extern "C" {
#include <ucp/api/device/ucp_host.h>
}

class nixlUcxEp;

namespace nixl::ucx {
class rkey;

class deviceMemList {
public:
    deviceMemList() = delete;
    deviceMemList(const nixlUcxEp &ep,
                  const std::vector<ucp_mem_h> &local_memhs,
                  const std::vector<const nixl::ucx::rkey *> &remote_rkeys);
    explicit deviceMemList(const ucp_device_mem_list_handle_h) noexcept;

    [[nodiscard]] ucp_device_mem_list_handle_h
    get() const noexcept {
        return deviceMemList_.get();
    }

private:
    [[nodiscard]] static ucp_device_mem_list_handle_h
    createDeviceMemList(const nixlUcxEp &ep,
                       const std::vector<ucp_mem_h> &local_memhs,
                       const std::vector<const nixl::ucx::rkey *> &remote_rkeys);

    const std::unique_ptr<ucp_device_mem_list_handle, void (*)(ucp_device_mem_list_handle_h)>
        deviceMemList_;
};
} // namespace nixl::ucx

#endif
