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

#include "device_mem_list.h"

#include <stdexcept>

#include "common/nixl_log.h"
#include "ucx_utils.h"
#include "rkey.h"
#include "config.h"

namespace nixl::ucx {

deviceMemList::deviceMemList(const nixlUcxEp &ep,
                             const std::vector<ucp_mem_h> &local_memhs,
                             const std::vector<const nixl::ucx::rkey *> &remote_rkeys)
    : deviceMemList_{createDeviceMemList(ep, local_memhs, remote_rkeys), &ucp_device_mem_list_release} {}

deviceMemList::deviceMemList(const ucp_device_mem_list_handle_h device_mem_list) noexcept
    : deviceMemList_{device_mem_list, &ucp_device_mem_list_release} {}

ucp_device_mem_list_handle_h
deviceMemList::createDeviceMemList(const nixlUcxEp &ep,
                                   const std::vector<ucp_mem_h> &local_memhs,
                                   const std::vector<const nixl::ucx::rkey *> &remote_rkeys) {
    nixl_status_t status = ep.checkTxState();
    if (status != NIXL_SUCCESS) {
        throw std::runtime_error("Endpoint not in valid state for creating memory list");
    }

    if (local_memhs.empty() || remote_rkeys.empty()) {
        throw std::invalid_argument("Empty memh or rkey lists provided");
    }

    if (local_memhs.size() != remote_rkeys.size()) {
        throw std::invalid_argument("Local memh and remote rkey lists must have same size");
    }

    std::vector<ucp_device_mem_list_elem_t> ucp_elements;
    ucp_elements.reserve(local_memhs.size());

    for (size_t i = 0; i < local_memhs.size(); i++) {
        ucp_device_mem_list_elem_t ucp_elem;
        ucp_elem.field_mask =
            UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH | UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY;
        ucp_elem.memh = local_memhs[i];
        ucp_elem.rkey = remote_rkeys[i]->get();
        ucp_elements.push_back(ucp_elem);
    }

    ucp_device_mem_list_params_t params;
    params.field_mask = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
        UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
        UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
    params.elements = ucp_elements.data();
    params.element_size = sizeof(ucp_device_mem_list_elem_t);
    params.num_elements = ucp_elements.size();

    ucp_device_mem_list_handle_h ucx_handle;
    ucs_status_t ucs_status = ucp_device_mem_list_create(ep.getEp(), &params, &ucx_handle);
    if (ucs_status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to create device memory list: ") +
                                 ucs_status_string(ucs_status));
    }

    NIXL_DEBUG << "Created device memory list handle with " << local_memhs.size() << " elements";
    return ucx_handle;
}

} // namespace nixl::ucx
