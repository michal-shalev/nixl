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
#include "ucx_utils.h"

#include <exception>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include <nixl_types.h>

#include "common/nixl_log.h"
#include "config.h"
#include "serdes/serdes.h"

extern "C"
{
#include <ucp/api/ucp.h>
}

using namespace std;

nixl_status_t ucx_status_to_nixl(ucs_status_t status)
{
    if (status == UCS_OK) {
        return NIXL_SUCCESS;
    }

    switch(status) {
    case UCS_INPROGRESS:
        return NIXL_IN_PROG;
    case UCS_ERR_NOT_CONNECTED:
    case UCS_ERR_CONNECTION_RESET:
    case UCS_ERR_ENDPOINT_TIMEOUT:
        return NIXL_ERR_REMOTE_DISCONNECT;
    case UCS_ERR_INVALID_PARAM:
        return NIXL_ERR_INVALID_PARAM;
    default:
        NIXL_WARN << "Unexpected UCX error: " << ucs_status_string(status);
        return NIXL_ERR_BACKEND;
    }
}

static void err_cb_wrapper(void *arg, ucp_ep_h ucp_ep, ucs_status_t status)
{
    nixlUcxEp *ep = reinterpret_cast<nixlUcxEp*>(arg);
    ep->err_cb(ucp_ep, status);
}

void nixlUcxEp::err_cb(ucp_ep_h ucp_ep, ucs_status_t status)
{
    ucs_status_ptr_t request;

    NIXL_DEBUG << "ep " << eph << ": state " << state
               << ", UCX error handling callback was invoked with status "
               << status << " (" << ucs_status_string(status) << ")";

    NIXL_ASSERT(eph == ucp_ep);

    switch(state) {
    case NIXL_UCX_EP_STATE_NULL:
    case NIXL_UCX_EP_STATE_FAILED:
        // The error was already handled, nothing to do
    case NIXL_UCX_EP_STATE_DISCONNECTED:
        // The EP has been disconnected, nothing to do
        return;
    case NIXL_UCX_EP_STATE_CONNECTED:
        setState(NIXL_UCX_EP_STATE_FAILED);
        request = ucp_ep_close_nb(ucp_ep, UCP_EP_CLOSE_MODE_FORCE);
        if (UCS_PTR_IS_PTR(request)) {
            ucp_request_free(request);
        }
        return;
    }
    NIXL_FATAL << "Invalid endpoint state: " << state;
    std::terminate();
}

void nixlUcxEp::setState(nixl_ucx_ep_state_t new_state)
{
    NIXL_ASSERT(new_state != state);
    NIXL_DEBUG << "ep " << eph << ": state " << state << " -> " << new_state;
    state = new_state;
}

nixl_status_t
nixlUcxEp::closeImpl(ucp_ep_close_flags_t flags)
{
    ucs_status_ptr_t request      = nullptr;
    ucp_request_param_t req_param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
        .flags        = flags
    };

    switch(state) {
    case NIXL_UCX_EP_STATE_NULL:
    case NIXL_UCX_EP_STATE_DISCONNECTED:
        // The EP has not been connected, or already disconnected.
        // Nothing to do.
        NIXL_ASSERT(eph == nullptr);
        return NIXL_SUCCESS;
    case NIXL_UCX_EP_STATE_FAILED:
        // The EP was closed in error callback, just return error.
        eph = nullptr;
        return NIXL_ERR_REMOTE_DISCONNECT;
    case NIXL_UCX_EP_STATE_CONNECTED:
        request = ucp_ep_close_nbx(eph, &req_param);
        if (request == nullptr) {
            eph = nullptr;
            return NIXL_SUCCESS;
        }

        if (UCS_PTR_IS_ERR(request)) {
            eph = nullptr;
            return ucx_status_to_nixl(UCS_PTR_STATUS(request));
        }

        ucp_request_free(request);
        eph = nullptr;
        return NIXL_SUCCESS;
    }
    NIXL_FATAL << "Invalid endpoint state: " << state;
    std::terminate();
}

nixlUcxEp::nixlUcxEp(ucp_worker_h worker, void* addr,
                     ucp_err_handling_mode_t err_handling_mode)
{
    ucp_ep_params_t ep_params;
    nixl_status_t status;

    ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode        = err_handling_mode;
    ep_params.err_handler.cb  = err_cb_wrapper;
    ep_params.err_handler.arg = reinterpret_cast<void*>(this);
    ep_params.address         = reinterpret_cast<ucp_address_t*>(addr);

    status = ucx_status_to_nixl(ucp_ep_create(worker, &ep_params, &eph));
    if (status == NIXL_SUCCESS)
        setState(NIXL_UCX_EP_STATE_CONNECTED);
    else
        throw std::runtime_error("failed to create ep");
}

 nixlUcxEp::~nixlUcxEp()
 {
     nixl_status_t status = disconnect_nb();
     if (status)
         NIXL_ERROR << "Failed to disconnect ep with status " << status;
 }

/* ===========================================
 * EP management
 * =========================================== */

nixl_status_t nixlUcxEp::disconnect_nb()
{
    nixl_status_t status = closeImpl(ucp_ep_close_flags_t(0));

    // At step of disconnect we can ignore the remote disconnect error.
    return (status == NIXL_ERR_REMOTE_DISCONNECT) ? NIXL_SUCCESS : status;
}

/* ===========================================
 * RKey management
 * =========================================== */

int nixlUcxEp::rkeyImport(void* addr, size_t size, nixlUcxRkey &rkey)
{
    ucs_status_t status;

    status = ucp_ep_rkey_unpack(eph, addr, &rkey.rkeyh);
    if (status != UCS_OK)
    {
        /* TODO: MSW_NET_ERROR(priv->net, "unable to unpack key!\n"); */
        return -1;
    }

    return 0;
}

void nixlUcxEp::rkeyDestroy(nixlUcxRkey &rkey)
{
    ucp_rkey_destroy(rkey.rkeyh);
}

/* ===========================================
 * Active message handling
 * =========================================== */

nixl_status_t nixlUcxEp::sendAm(unsigned msg_id,
                                void* hdr, size_t hdr_len,
                                void* buffer, size_t len,
                                uint32_t flags, nixlUcxReq &req)
{
    ucs_status_ptr_t request;
    ucp_request_param_t param = {0};

    param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
    param.flags         = flags;

    request = ucp_am_send_nbx(eph, msg_id, hdr, hdr_len, buffer, len, &param);

    if (UCS_PTR_IS_PTR(request)) {
        req = (void*)request;
        return NIXL_IN_PROG;
    }

    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

/* ===========================================
 * Data transfer
 * =========================================== */

nixl_status_t nixlUcxEp::read(uint64_t raddr, nixlUcxRkey &rk,
                              void *laddr, nixlUcxMem &mem,
                              size_t size, nixlUcxReq &req)
{
    nixl_status_t status = checkTxState();
    if (status != NIXL_SUCCESS) {
        return status;
    }

    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_MEMH |
                        UCP_OP_ATTR_FLAG_MULTI_SEND,
        .memh         = mem.memh,
    };

    ucs_status_ptr_t request = ucp_get_nbx(eph, laddr, size, raddr,
                                           rk.rkeyh, &param);
    if (UCS_PTR_IS_PTR(request)) {
        req = (void*)request;
        return NIXL_IN_PROG;
    }

    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

nixl_status_t nixlUcxEp::write(void *laddr, nixlUcxMem &mem,
                               uint64_t raddr, nixlUcxRkey &rk,
                               size_t size, nixlUcxReq &req)
{
    nixl_status_t status = checkTxState();
    if (status != NIXL_SUCCESS) {
        return status;
    }

    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_MEMH |
                        UCP_OP_ATTR_FLAG_MULTI_SEND,
        .memh         = mem.memh,
    };

    ucs_status_ptr_t request = ucp_put_nbx(eph, laddr, size, raddr,
                                           rk.rkeyh, &param);
    if (UCS_PTR_IS_PTR(request)) {
        req = (void*)request;
        return NIXL_IN_PROG;
    }

    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

nixl_status_t nixlUcxEp::estimateCost(size_t size,
                                      std::chrono::microseconds &duration,
                                      std::chrono::microseconds &err_margin,
                                      nixl_cost_t &method)
{
    ucp_ep_evaluate_perf_param_t params = {
        .field_mask   = UCP_EP_PERF_PARAM_FIELD_MESSAGE_SIZE,
        .message_size = size,
    };

    ucp_ep_evaluate_perf_attr_t cost_result = {
        .field_mask = UCP_EP_PERF_ATTR_FIELD_ESTIMATED_TIME,
    };

    ucs_status_t status = ucp_ep_evaluate_perf(this->eph, &params, &cost_result);
    if (status != UCS_OK) {
        NIXL_ERROR << "ucp_ep_evaluate_perf failed: " << ucs_status_string(status);
        return NIXL_ERR_BACKEND;
    }

    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<double>(cost_result.estimated_time));
    method = nixl_cost_t::ANALYTICAL_BACKEND;
    // Currently, we do not have a way to estimate the error margin
    err_margin = std::chrono::microseconds(0);
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEp::flushEp(nixlUcxReq &req)
{
    ucp_request_param_t param;
    ucs_status_ptr_t request;

    param.op_attr_mask = 0;
    request = ucp_ep_flush_nbx(eph, &param);

    if (UCS_PTR_IS_PTR(request)) {
        req = (void*)request;
        return NIXL_IN_PROG;
    }

    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

bool nixlUcxMtLevelIsSupported(const nixl_ucx_mt_t mt_type) noexcept
{
    ucp_lib_attr_t attr;
    attr.field_mask = UCP_LIB_ATTR_FIELD_MAX_THREAD_LEVEL;
    ucp_lib_query(&attr);

    switch(mt_type) {
    case nixl_ucx_mt_t::SINGLE:
        return attr.max_thread_level >= UCS_THREAD_MODE_SERIALIZED;
    case nixl_ucx_mt_t::CTX:
    case nixl_ucx_mt_t::WORKER:
        return attr.max_thread_level >= UCS_THREAD_MODE_MULTI;
    }
    NIXL_FATAL << "invalid mt type: " << enumToInteger(mt_type);
    std::terminate();
}

nixlUcxContext::nixlUcxContext(std::vector<std::string> devs,
                               size_t req_size,
                               nixlUcxContext::req_cb_t init_cb,
                               nixlUcxContext::req_cb_t fini_cb,
                               bool prog_thread,
                               ucp_err_handling_mode_t __err_handling_mode,
                               unsigned long num_workers,
                               nixl_thread_sync_t sync_mode)
{
    ucp_params_t ucp_params;

    // With strict synchronization model nixlAgent serializes access to backends, with more
    // permissive models backends need to account for concurrent access and ensure their internal
    // state is properly protected. Progress thread creates internal concurrency in UCX backend
    // irrespective of nixlAgent synchronization model.
    mt_type = (sync_mode == nixl_thread_sync_t::NIXL_THREAD_SYNC_RW || prog_thread) ?
        nixl_ucx_mt_t::WORKER : nixl_ucx_mt_t::SINGLE;
    err_handling_mode = __err_handling_mode;

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    ucp_params.features = UCP_FEATURE_RMA | UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64 | UCP_FEATURE_AM;
    if (prog_thread)
        ucp_params.features |= UCP_FEATURE_WAKEUP;
    ucp_params.mt_workers_shared = num_workers > 1 ? 1 : 0;

    if (req_size) {
        ucp_params.request_size = req_size;
        ucp_params.field_mask |= UCP_PARAM_FIELD_REQUEST_SIZE;
    }

    if (init_cb) {
        ucp_params.request_init = init_cb;
        ucp_params.field_mask |= UCP_PARAM_FIELD_REQUEST_INIT;
    }

    if (fini_cb) {
        ucp_params.request_cleanup = fini_cb;
        ucp_params.field_mask |= UCP_PARAM_FIELD_REQUEST_CLEANUP;
    }

    nixl::ucx::config config;

    /* If requested, restrict the set of network devices */
    if (devs.size()) {
        /* TODO: check if this is the best way */
        std::string devs_str;
        for (const auto &dev : devs) {
            devs_str += dev + ":1,";
        }
        devs_str.pop_back(); // to remove odd comma after the last device
        config.modifyAlways ("NET_DEVICES", devs_str.c_str());
    }

    unsigned major_version, minor_version, release_number;
    ucp_get_version(&major_version, &minor_version, &release_number);

    config.modify ("ADDRESS_VERSION", "v2");
    config.modify ("RNDV_THRESH", "inf");

    unsigned ucp_version = UCP_VERSION(major_version, minor_version);
    if (ucp_version >= UCP_VERSION(1, 19)) {
        config.modify ("MAX_COMPONENT_MDS", "32");
    }

    if (ucp_version >= UCP_VERSION(1, 20)) {
        config.modify ("MAX_RMA_RAILS", "4");
    } else {
        config.modify ("MAX_RMA_RAILS", "2");
    }

    const auto status = ucp_init (&ucp_params, config.getUcpConfig(), &ctx);
    if (status != UCS_OK) {
        throw std::runtime_error ("Failed to create UCX context: " +
                                  std::string (ucs_status_string (status)));
    }
}

nixlUcxContext::~nixlUcxContext()
{
    ucp_cleanup(ctx);
}

namespace
{
   [[nodiscard]] ucs_thread_mode_t toUcsThreadModeChecked(const nixl_ucx_mt_t t)
   {
       switch(t) {
           case nixl_ucx_mt_t::CTX:
               return UCS_THREAD_MODE_SINGLE;
           case nixl_ucx_mt_t::SINGLE:
               return UCS_THREAD_MODE_SERIALIZED;
           case nixl_ucx_mt_t::WORKER:
               return UCS_THREAD_MODE_MULTI;
       }
       NIXL_FATAL << "Invalid UCX worker type: " << static_cast<std::underlying_type_t<nixl_ucx_mt_t>>(t);
       std::terminate();
   }

   struct nixlUcpWorkerParams
       : ucp_worker_params_t
   {
       explicit nixlUcpWorkerParams(const nixl_ucx_mt_t t)
       {
           field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
           thread_mode = toUcsThreadModeChecked(t);
       }

       nixlUcpWorkerParams &withRmaBatchCallback(ucp_rma_batch_callback_t cb, void *arg) {
            field_mask |= UCP_WORKER_PARAM_FIELD_RMA_BATCH_CB;
            rma_batch_callback = cb;
            if (arg) {
                field_mask |= UCP_WORKER_PARAM_FIELD_RMA_BATCH_CB_ARG;
                rma_batch_cb_arg = arg;
            }
            return *this;
        }
   };

}  // namespace

ucp_worker* nixlUcxWorker::createUcpWorker(nixlUcxContext& ctx, ucp_rma_batch_callback_t callback, void* callback_arg)
{
    ucp_worker* worker = nullptr;

    nixlUcpWorkerParams params(ctx.mt_type);
    if (callback) {
        params.withRmaBatchCallback(callback, callback_arg);
    }

    const ucs_status_t status = ucp_worker_create(ctx.ctx, &params, &worker);
    if(status != UCS_OK) {
        const auto err_str = std::string("Failed to create UCX worker: ") +
                             ucs_status_string(status);
        NIXL_ERROR << err_str;
        throw std::runtime_error(err_str);  // worker is nullptr -- no leak
    }
    return worker;
}

nixlUcxWorker::nixlUcxWorker(const std::shared_ptr< nixlUcxContext >&_ctx, ucp_rma_batch_callback_t rma_batch_callback, void* callback_arg)
    : ctx(_ctx),
      worker(createUcpWorker(*ctx, rma_batch_callback, callback_arg), &ucp_worker_destroy)
{}

std::string nixlUcxWorker::epAddr()
{
    ucp_worker_attr_t wattr;

    wattr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
    const ucs_status_t status = ucp_worker_query(worker.get(), &wattr);
    if (UCS_OK != status) {
        NIXL_WARN << "Unable to query UCX endpoint address";
        return {};
    }
    const std::string result = nixlSerDes::_bytesToString(wattr.address, wattr.address_length);
    ucp_worker_release_address(worker.get(), wattr.address);
    return result;
}

absl::StatusOr<std::unique_ptr<nixlUcxEp>> nixlUcxWorker::connect(void* addr, std::size_t size)
{
    try {
        return std::make_unique<nixlUcxEp>(worker.get(), addr, ctx->err_handling_mode);
    } catch (const std::exception &e) {
        return absl::UnavailableError(e.what());
    }
}

/* ===========================================
 * Memory management
 * =========================================== */


int nixlUcxContext::memReg(void *addr, size_t size, nixlUcxMem &mem, nixl_mem_t nixl_mem_type)
{
    //mem.uw = this;
    mem.base = addr;
    mem.size = size;

    ucp_mem_map_params_t mem_params = {
        .field_mask = UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                     UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                     UCP_MEM_MAP_PARAM_FIELD_ADDRESS,
        .address = mem.base,
        .length  = mem.size,
    };

    ucs_status_t status = ucp_mem_map(ctx, &mem_params, &mem.memh);
    if (status != UCS_OK) {
        /* TODOL: MSW_NET_ERROR(priv->net, "failed to ucp_mem_map (%s)\n", ucs_status_string(status)); */
        return -1;
    }

    if (nixl_mem_type == nixl_mem_t::VRAM_SEG) {
        ucp_mem_attr_t attr;
        attr.field_mask = UCP_MEM_ATTR_FIELD_MEM_TYPE;
        status = ucp_mem_query(mem.memh, &attr);
        if (status != UCS_OK) {
            NIXL_ERROR << absl::StrFormat("Failed to ucp_mem_query: %s",
                                          ucs_status_string(status));
            ucp_mem_unmap(ctx, mem.memh);
            return -1;
        }

        if (attr.mem_type == UCS_MEMORY_TYPE_HOST) {
            NIXL_ERROR << "memory is detected as host, check that UCX is configured"
                          " with CUDA support";
            ucp_mem_unmap(ctx, mem.memh);
            return -1;
        }
    }

    return 0;
}

std::string nixlUcxContext::packRkey(nixlUcxMem &mem)
{
    void* rkey_buf;
    std::size_t size;

    const ucs_status_t status = ucp_rkey_pack(ctx, mem.memh, &rkey_buf, &size);
    if (status != UCS_OK) {
        /* TODO: MSW_NET_ERROR(priv->net, "failed to ucp_rkey_pack (%s)\n", ucs_status_string(status)); */
        return {};
    }
    const std::string result = nixlSerDes::_bytesToString(rkey_buf, size);
    ucp_rkey_buffer_release(rkey_buf);
    return result;
}

void nixlUcxContext::memDereg(nixlUcxMem &mem)
{
    ucp_mem_unmap(ctx, mem.memh);
}

/* ===========================================
 * Active message handling
 * =========================================== */

int nixlUcxWorker::regAmCallback(unsigned msg_id, ucp_am_recv_callback_t cb, void* arg)
{
    ucp_am_handler_param_t params = {0};

    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;

    params.id = msg_id;
    params.cb = cb;
    params.arg = arg;

    const ucs_status_t status = ucp_worker_set_am_recv_handler(worker.get(), &params);

    if(status != UCS_OK) {
        //TODO: error handling
        return -1;
    }
    return 0;
}

/* ===========================================
 * Data transfer
 * =========================================== */

int nixlUcxWorker::progress()
{
  return ucp_worker_progress(worker.get());
}

nixl_status_t nixlUcxWorker::test(nixlUcxReq req)
{
    if(req == nullptr) {
        return NIXL_SUCCESS;
    }
    ucp_worker_progress(worker.get());
    return ucx_status_to_nixl(ucp_request_check_status(req));
}

void nixlUcxWorker::reqRelease(nixlUcxReq req)
{
    ucp_request_free((void*)req);
}

void nixlUcxWorker::reqCancel(nixlUcxReq req)
{
    ucp_request_cancel(worker.get(), req);
}

nixl_status_t nixlUcxEp::prepareBatch(const std::vector<std::tuple<uint64_t,
                                      void*, nixlUcxRkey&, nixlUcxMem&,
                                      size_t>>& operations,
                                      bool has_completion_msg,
                                      const std::string& completion_msg,
                                      nixlUcxReq &req)
{
    nixl_status_t status = checkTxState();
    if (status != NIXL_SUCCESS) {
        return status;
    }

    std::vector<ucp_rma_batch_iov_elem_t> batch_iov(operations.size());
    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& [remote_addr, local_addr, rkey, mem, size] = operations[i];

        uintptr_t local_ptr  = reinterpret_cast<uintptr_t>(local_addr);
        uintptr_t remote_ptr = static_cast<uintptr_t>(remote_addr);

        if (local_ptr % 8 != 0 || remote_ptr % 8 != 0) {
            NIXL_ERROR << "Unaligned access: local=" << local_ptr
                    << " remote=" << remote_ptr << " index=" << i;
        }

        batch_iov[i].opcode     = UCP_OP_PUT;
        batch_iov[i].local_va   = local_addr;
        batch_iov[i].memh       = mem.memh;
        batch_iov[i].remote_va  = remote_addr;
        batch_iov[i].rkey       = rkey.rkeyh;
        batch_iov[i].length     = size;
    }

    ucp_rma_batch_param_t param = {};
    if (has_completion_msg) {
        param.field_mask = UCP_RMA_BATCH_FIELD_COMPLETION_MESSAGE |
                          UCP_RMA_BATCH_FIELD_COMPLETION_MESSAGE_LENGTH;
        param.completion_message = const_cast<void*>(static_cast<const void*>(completion_msg.c_str()));
        param.completion_message_length = completion_msg.size();
    }

    ucs_status_ptr_t request = ucp_ep_rma_prepare_batch(eph, batch_iov.data(), operations.size(),
                                                        has_completion_msg ? &param : nullptr);
    if (UCS_PTR_IS_PTR(request)) {
        req = (void*)request;
        return NIXL_IN_PROG;
    }

    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}

nixl_status_t nixlUcxEp::postBatch(nixlUcxReq batch_req)
{
    nixl_status_t status = checkTxState();
    if (status != NIXL_SUCCESS) {
        return status;
    }

    ucs_status_ptr_t request = ucp_ep_rma_post_batch(batch_req);
    if (UCS_PTR_IS_PTR(request)) {
        // For now, we'll just ignore the post request handle
        // In a full implementation, this might need to be tracked
        return NIXL_IN_PROG;
    }

    return ucx_status_to_nixl(UCS_PTR_STATUS(request));
}
