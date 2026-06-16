/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucx_backend.h"
#include "common/nixl_log.h"
#include "serdes/serdes.h"
#include "common/nixl_log.h"

#include <optional>
#include <limits>
#include <future>
#include <set>
#include <string.h>
#include <unistd.h>
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include <asio.hpp>
namespace {
    void moveNotifList(notif_list_t &src, notif_list_t &tgt)
    {
        if (src.size() > 0) {
            std::move(src.begin(), src.end(), std::back_inserter(tgt));
            src.clear();
        }
    }
}

/****************************************
 * Completion state (callback-based completion notification)
 *****************************************/

/*
 * Per-transfer completion state shared between the posting/waiting thread and
 * the UCX completion callbacks (which may run from the progress thread or from
 * whichever thread drives ucp_worker_progress()).
 *
 * Only atomics are touched from the callback: a completed counter, and on
 * failure a single status field set via compare-exchange. No allocation, no
 * locks, no Python, and no UCX progress are performed in the callback.
 */
struct nixlUcxCompletionState {
    // Number of callbacks we expect to fire. Incremented via expect() *before*
    // the UCX op is submitted, so a callback running on the progress thread can
    // never be observed before it has been accounted for. Decremented via
    // unexpect() when an op completes without a callback (immediate path).
    std::atomic<size_t> expected{0};
    // Number of completion callbacks observed so far.
    std::atomic<size_t> completed{0};
    // Immediate (no-callback) completions, kept separately for debugging.
    std::atomic<size_t> immediate{0};
    // First failure status reported by any callback (NIXL_SUCCESS if none).
    std::atomic<nixl_status_t> status{NIXL_SUCCESS};

    nixlUcxCompletionState() = default;

    // Atomics are not movable by default, but the enclosing handle is stored in
    // a std::vector (composite chunks) that must stay movable. These are only
    // invoked at handle (re)construction, before any request is posted, so a
    // plain load/store relocation is safe and race-free.
    nixlUcxCompletionState(nixlUcxCompletionState &&other) noexcept
        : expected(other.expected.load(std::memory_order_relaxed)),
          completed(other.completed.load(std::memory_order_relaxed)),
          immediate(other.immediate.load(std::memory_order_relaxed)),
          status(other.status.load(std::memory_order_relaxed)) {}

    nixlUcxCompletionState &
    operator=(nixlUcxCompletionState &&other) noexcept {
        expected.store(other.expected.load(std::memory_order_relaxed), std::memory_order_relaxed);
        completed.store(other.completed.load(std::memory_order_relaxed), std::memory_order_relaxed);
        immediate.store(other.immediate.load(std::memory_order_relaxed), std::memory_order_relaxed);
        status.store(other.status.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }

    void
    reset() noexcept {
        expected.store(0, std::memory_order_relaxed);
        completed.store(0, std::memory_order_relaxed);
        immediate.store(0, std::memory_order_relaxed);
        status.store(NIXL_SUCCESS, std::memory_order_relaxed);
    }

    // Register an expected completion callback. MUST be called before the UCX
    // op is submitted so the callback (which may run immediately on another
    // thread) is always accounted for before it can fire.
    void
    expect() noexcept {
        expected.fetch_add(1, std::memory_order_relaxed);
    }

    // Roll back an expectation for an op that completed without a callback
    // (immediate UCS_OK or immediate error).
    void
    unexpect() noexcept {
        expected.fetch_sub(1, std::memory_order_relaxed);
    }

    // Account an immediate (no-callback) completion, for debugging.
    void
    recordImmediate() noexcept {
        immediate.fetch_add(1, std::memory_order_relaxed);
    }

    // Called from the completion callback when a posted op finishes.
    void
    complete(ucs_status_t s) noexcept {
        if (__builtin_expect(s != UCS_OK, 0)) {
            recordFailure(ucx_status_to_nixl(s));
        }
        completed.fetch_add(1, std::memory_order_release);
    }

    // True once every expected callback has fired. expect() runs before submit,
    // so completed never exceeds expected and done() is never true early.
    [[nodiscard]] bool
    done() const noexcept {
        return completed.load(std::memory_order_acquire) >=
               expected.load(std::memory_order_acquire);
    }

    void
    recordFailure(nixl_status_t s) noexcept {
        nixl_status_t expected_status = NIXL_SUCCESS;
        status.compare_exchange_strong(expected_status, s, std::memory_order_acq_rel);
    }
};

/*
 * UCX completion callback attached to every posted RMA/flush request. Minimal
 * by design: record status/bump the completed counter (complete()), then free
 * the request.
 *
 * Freeing here is the key lifetime rule: the request is released only when its
 * callback actually fires (i.e. after completion is observed), never before.
 * This is the canonical UCX idiom and avoids both (a) retaining O(N) requests
 * until end-of-transfer and (b) the "free-before-completion" pattern that
 * silently suppresses callbacks.
 */
static void
nixlUcxXferCompletionCb(void *request, ucs_status_t status, void *user_data) {
    auto *state = static_cast<nixlUcxCompletionState *>(user_data);
    state->complete(status);
    ucp_request_free(request);
}

/****************************************
 * Backend request management
*****************************************/

class nixlUcxBackendH : public nixlBackendReqH {
private:
    std::set<ucx_connection_ptr_t> connections_;
    std::vector<nixlUcxReq> requests_;
    nixlUcxCompletionState completion_;
    nixlUcxWorker *worker;
    size_t worker_id;
    // Whether the waiting/releasing thread should drive ucp_worker_progress().
    // False when an external UCX progress thread owns the worker, so we only
    // observe completion via the atomic counters / request status.
    bool pollProgress_;

    // Notification to be sent after completion of all requests
    struct Notif {
        std::string agent;
        nixl_blob_t payload;

        Notif(const std::string &remote_agent, const nixl_blob_t &msg)
            : agent(remote_agent),
              payload(msg) {}
    };
    std::optional<Notif> notif;

    nixl_status_t
    checkConnection(nixl_status_t status = NIXL_SUCCESS) const {
        NIXL_ASSERT(!connections_.empty());
        for (const auto &conn : connections_) {
            nixl_status_t conn_status = conn->getEp(worker_id)->checkTxState();
            if (conn_status != NIXL_SUCCESS) {
                return conn_status;
            }
        }
        return status;
    }

public:
    nixlUcxBackendH(nixlUcxWorker *worker, size_t worker_id, bool poll_progress = true)
        : worker(worker),
          worker_id(worker_id),
          pollProgress_(poll_progress) {}

    auto &
    notification() {
        return notif;
    }

    // Completion state shared with the UCX completion callbacks. Stable for the
    // lifetime of the handle, so the pointer can be safely handed to UCX as
    // user_data for every request posted from this handle.
    nixlUcxCompletionState *
    completionState() noexcept {
        return &completion_;
    }

    void
    reserve(size_t size) {
        requests_.reserve(size);
        NIXL_ASSERT(connections_.empty());
        completion_.reset();
    }

    // Record a distinct connection once per EP batch (not per request), so the
    // flush phase can iterate over all touched connections.
    void
    addConnection(const ucx_connection_ptr_t &conn) {
        connections_.insert(conn);
    }

    // Account for a callback-tracked request (RMA / flush) after submission.
    // The matching completion_.expect() must already have been called before
    // the op was submitted. The request itself is owned by its completion
    // callback (which frees it), so nothing is stored here.
    nixl_status_t
    append(nixl_status_t status) {
        switch (status) {
        case NIXL_IN_PROG:
            // A completion callback will fire (and free the request) later;
            // the expectation registered before submit stands.
            break;
        case NIXL_SUCCESS:
            // Immediate completion: UCX returned UCS_OK, no callback will fire,
            // so roll back the expectation and account it separately.
            completion_.unexpect();
            completion_.recordImmediate();
            break;
        default:
            // Immediate error: no callback will fire. Roll back the expectation
            // and release all previously initiated ops.
            completion_.unexpect();
            release();
            return status;
        }
        return NIXL_SUCCESS;
    }

    // Append a request that is NOT callback-tracked (notification sendAm, which
    // carries its own AM callback). These keep the original request-polling
    // lifetime: stored here, polled in status(), freed when complete.
    nixl_status_t
    appendLegacy(nixl_status_t status, nixlUcxReq req, const ucx_connection_ptr_t &conn) {
        switch (status) {
        case NIXL_IN_PROG:
            requests_.push_back(req);
            connections_.insert(conn);
            break;
        case NIXL_SUCCESS:
            connections_.insert(conn);
            break;
        default:
            release();
            return status;
        }
        return NIXL_SUCCESS;
    }

    const std::set<ucx_connection_ptr_t> &
    getConnections() const {
        return connections_;
    }

    virtual bool
    isComposite() const {
        return false;
    }

    virtual nixl_status_t
    release() {
        // Callback-tracked requests free themselves from their completion
        // callback, which dereferences completion_ (this handle's member). We
        // must not let the handle be destroyed while any such callback is still
        // outstanding, so drain them first.
        //
        // The drain is bounded and stall-aware so it can never spin forever:
        // the deadline is reset whenever a completion is observed, so a slow
        // but progressing transfer is tolerated, while a genuinely stuck UCX
        // state (no forward progress) is abandoned with a loud error. When an
        // external progress thread owns the worker, we wait (yield) instead of
        // calling progress() ourselves.
        nixl_status_t drain_ret = NIXL_SUCCESS;
        if (!completion_.done()) {
            constexpr auto stall_timeout = std::chrono::seconds(10);
            size_t last_completed = completion_.completed.load(std::memory_order_acquire);
            auto deadline = std::chrono::steady_clock::now() + stall_timeout;

            while (!completion_.done()) {
                if (pollProgress_) {
                    worker->progress();
                } else {
                    std::this_thread::yield();
                }

                const size_t cur = completion_.completed.load(std::memory_order_acquire);
                if (cur != last_completed) {
                    // Forward progress: extend the deadline.
                    last_completed = cur;
                    deadline = std::chrono::steady_clock::now() + stall_timeout;
                } else if (std::chrono::steady_clock::now() >= deadline) {
                    NIXL_ERROR << "UCX release() stalled: "
                               << (completion_.expected.load() - cur)
                               << " completion callback(s) never fired; abandoning drain";
                    drain_ret = NIXL_ERR_BACKEND;
                    break;
                }
            }
        }

        // Legacy (notification) requests are owned here: cancel if still in
        // progress, then free.
        for (nixlUcxReq req : requests_) {
            nixl_status_t ret = ucx_status_to_nixl(ucp_request_check_status(req));
            if (ret == NIXL_IN_PROG) {
                worker->reqCancel(req);
            }
            worker->reqRelease(req);
        }
        requests_.clear();
        connections_.clear();
        completion_.reset();
        return drain_ret;
    }

    virtual nixl_status_t
    status() {
        /* Drive progress only when no external progress thread owns the worker;
         * otherwise that thread fires the completion callbacks and we just read
         * the atomic counters (and poll any legacy notification request). */
        if (pollProgress_) {
            while (worker->progress())
                ;
        }

        const bool data_complete = completion_.done();

        /* Poll legacy (notification) requests, if any. */
        bool legacy_complete = true;
        if (!requests_.empty()) {
            size_t incomplete_reqs = 0;
            for (nixlUcxReq req : requests_) {
                nixl_status_t ret = ucx_status_to_nixl(ucp_request_check_status(req));
                if (ret == NIXL_SUCCESS) {
                    worker->reqRelease(req);
                } else if (ret == NIXL_IN_PROG) {
                    legacy_complete = false;
                    requests_[incomplete_reqs++] = req;
                } else {
                    requests_.resize(incomplete_reqs);
                    return checkConnection(ret);
                }
            }
            requests_.resize(incomplete_reqs);
        }

        if (!data_complete || !legacy_complete) {
            return NIXL_IN_PROG;
        }

        /* All callbacks observed and legacy requests drained. */
        nixl_status_t out_ret = completion_.status.load(std::memory_order_acquire);
        connections_.clear();

        return (out_ret == NIXL_SUCCESS) ? NIXL_SUCCESS : checkConnection(out_ret);
    }

    void
    setWorker(nixlUcxWorker *worker, size_t worker_id) {
        NIXL_ASSERT(this->worker == nullptr || worker == nullptr);
        this->worker = worker;
        this->worker_id = worker_id;
    }

    nixlUcxWorker *
    getWorker() const {
        return worker;
    }

    size_t getWorkerId() const {
        return worker_id;
    }
};

/****************************************
 * Progress thread management
*****************************************/

/*
 * This class encapsulates a thread that polls one or multiple UCX workers
 */
class nixlUcxThread {
public:
    nixlUcxThread(const nixlUcxEngine *engine, size_t num_workers) : engine_(engine) {
        workers_.reserve(num_workers);
    }

    virtual ~nixlUcxThread() {
        if (threadActive_) {
            join();
        }
    }

    void
    start() {
        NIXL_ASSERT(!threadActive_);
        threadActive_ = std::make_unique<std::promise<void>>();
        auto active = threadActive_->get_future();
        thread_ = std::make_unique<std::thread>(std::ref(*this));
        active.wait();
    }

    virtual void
    join() {
        NIXL_ASSERT(threadActive_);
        threadActive_.reset();
        thread_->join();
    }

    virtual void
    addWorker(nixlUcxWorker *worker, size_t worker_id) {
        NIXL_ASSERT(workers_.size() < workers_.capacity());
        workers_.push_back(worker);
        workerIds_.push_back(worker_id);
    }

    const std::vector<nixlUcxWorker *> &
    getWorkers() const {
        return workers_;
    }

    size_t
    getWorkerId(size_t idx = 0) const {
        return workerIds_[idx];
    }

    void
    operator()() {
        tlsThread() = this;
        threadActive_->set_value();
        run();
    }

    static nixlUcxThread *&
    tlsThread() {
        static thread_local nixlUcxThread *tls = nullptr;
        return tls;
    }

    static bool
    isProgressThread(const nixlUcxEngine *engine) noexcept {
        nixlUcxThread *thread = tlsThread();
        return thread && thread->engine_ == engine;
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxThread &thread) {
        return os << "thread " << &thread << "{engine: " << thread.engine_ << ", worker_ids: ["
                  << absl::StrJoin(thread.workerIds_, ",") << "]}";
    }

protected:
    virtual void
    run() = 0;

private:
    const nixlUcxEngine *engine_;
    std::vector<nixlUcxWorker *> workers_;
    std::vector<size_t> workerIds_;
    std::unique_ptr<std::thread> thread_;
    std::unique_ptr<std::promise<void>> threadActive_;
};

class nixlUcxSharedThread : public nixlUcxThread {
public:
    nixlUcxSharedThread(const nixlUcxEngine *engine, size_t num_workers, nixlTime::us_t delay)
        : nixlUcxThread(engine, num_workers) {
        if (pipe(controlPipe_) < 0) {
            throw std::runtime_error("Couldn't create progress thread control pipe");
        }
        // TODO: We need delay to manual periodic wakeup/polling as a temporary
        // workaround for UCX bug (poll wouldn't wake up some fds in particular
        // circumstances)

        // This will ensure that the resulting delay is at least 1ms and fits into int in order for
        // it to be compatible with poll()
        int delay_us = std::min((int)delay, std::numeric_limits<int>::max());
        delay_ = std::chrono::ceil<std::chrono::milliseconds>(std::chrono::microseconds(delay_us));

        pollFds_.resize(num_workers + 1);
        pollFds_.back() = {controlPipe_[0], POLLIN, 0};
    }

    ~nixlUcxSharedThread() {
        close(controlPipe_[0]);
        close(controlPipe_[1]);
    }

    void
    join() override {
        const char signal = 'X';
        int ret = write(controlPipe_[1], &signal, sizeof(signal));
        if (ret < 0) NIXL_PERROR << "write to progress thread control pipe failed";
        nixlUcxThread::join();
    }

    void
    addWorker(nixlUcxWorker *worker, size_t worker_id) override {
        pollFds_[getWorkers().size()] = {worker->getEfd(), POLLIN, 0};
        nixlUcxThread::addWorker(worker, worker_id);
    }

protected:
    void
    run() override {
        NIXL_DEBUG << "shared " << *this << " running";
        // Set timeout event so that the main loop would progress all workers on first iteration
        bool timeout = true;
        bool pthr_stop = false;
        while (!pthr_stop) {
            for (size_t i = 0; i < pollFds_.size() - 1; i++) {
                if (!(pollFds_[i].revents & POLLIN) && !timeout) continue;
                pollFds_[i].revents = 0;
                nixlUcxWorker *worker = getWorkers()[i];
                do {
                    while (worker->progress())
                        ;
                } while (worker->arm() == NIXL_IN_PROG);
            }
            timeout = false;

            int ret;
            while ((ret = poll(pollFds_.data(), pollFds_.size(), delay_.count())) < 0)
                NIXL_PTRACE << "Call to poll() was interrupted, retrying";

            if (!ret) {
                timeout = true;
            } else if (pollFds_.back().revents & POLLIN) {
                pollFds_.back().revents = 0;

                char signal;
                int ret = read(pollFds_.back().fd, &signal, sizeof(signal));
                if (ret < 0) NIXL_PERROR << "read() on control pipe failed";

                pthr_stop = true;
            }
        }

        NIXL_DEBUG << "shared " << *this << " exiting";
    }

private:
    std::chrono::milliseconds delay_;
    int controlPipe_[2];
    std::vector<pollfd> pollFds_;
};

nixlUcxThreadEngine::nixlUcxThreadEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    if (!nixlUcxMtLevelIsSupported(nixl_ucx_mt_t::WORKER)) {
        throw std::invalid_argument("UCX library does not support multi-threading");
    }

    size_t num_workers = getWorkers().size();
    thread_ = std::make_unique<nixlUcxSharedThread>(this, num_workers, init_params.pthrDelay);
    for (size_t i = 0; i < num_workers; i++) {
        thread_->addWorker(getWorkers()[i].get(), i);
    }
    thread_->start();
}

nixlUcxThreadEngine::~nixlUcxThreadEngine() {
    thread_->join();
}

void
nixlUcxThreadEngine::appendNotif(std::string remote_name, std::string msg) {
    if (nixlUcxThread::isProgressThread(this)) {
        /* Append to the private list to allow batching */
        const std::lock_guard<std::mutex> lock(notifMtx_);
        notifPthr_.push_back(std::make_pair(std::move(remote_name), std::move(msg)));
    } else {
        nixlUcxEngine::appendNotif(std::move(remote_name), std::move(msg));
    }
}

nixl_status_t
nixlUcxThreadEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) return NIXL_ERR_INVALID_PARAM;

    getNotifsImpl(notif_list);
    const std::lock_guard<std::mutex> lock(notifMtx_);
    moveNotifList(notifPthr_, notif_list);
    return NIXL_SUCCESS;
}

/****************************************
 * Threadpool engine
 ****************************************/

struct nixlUcxBackendSharedState;

/*
 * This class represents a chunk of a composite request.
 * It is used to encapsulate a batch of requests (subset of the larger batch)
 * performed by a dedicated worker thread of threadpool. It holds a shared state
 * with the main request to track its completion status and control the lifetime.
 */
class nixlUcxChunkBackendH : public nixlUcxBackendH {
public:
    nixlUcxChunkBackendH() : nixlUcxBackendH(nullptr, UINT64_MAX) {}

    void
    startXfer(const std::shared_ptr<nixlUcxBackendSharedState> &shared_state,
              nixlUcxWorker *worker,
              size_t worker_id) {
        NIXL_ASSERT(sharedState_.get() == nullptr);
        sharedState_ = shared_state;
        setWorker(worker, worker_id);
    }

    void
    complete(nixl_status_t status);

    nixl_status_t
    status() override;

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxChunkBackendH &chunk) {
        return os << "chunk " << &chunk << "{worker_id: " << chunk.getWorkerId()
                  << ", state: " << chunk.sharedState_.get() << "}";
    }

private:
    std::shared_ptr<nixlUcxBackendSharedState> sharedState_;
};

/*
 * This class represents a shared state between a main request and all of its
 * chunks. It is used to track the completion status of the request and the
 * number of pending requests, and to control the lifetime of the chunks.
 */
struct nixlUcxBackendSharedState {
    std::atomic<nixl_status_t> status;
    std::atomic<size_t> pendingReqs;
    std::vector<nixlUcxChunkBackendH> chunks;

    nixlUcxBackendSharedState() : status(NIXL_SUCCESS), pendingReqs(0) {}

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxBackendSharedState &state) {
        return os << "state " << &state << "{status: " << state.status.load()
                  << ", pending=" << state.pendingReqs.load() << "}";
    }
};

void
nixlUcxChunkBackendH::complete(nixl_status_t status) {
    NIXL_ASSERT(sharedState_.get() != nullptr);
    if (status != NIXL_SUCCESS) {
        nixlUcxBackendH::release();
        sharedState_->status.store(status);
    }
    sharedState_->pendingReqs.fetch_sub(1);
    NIXL_TRACE << *this << " completed with status: " << status << ", " << *sharedState_;
    setWorker(nullptr, UINT64_MAX);
    sharedState_.reset();
}

nixl_status_t
nixlUcxChunkBackendH::status() {
    // First check if entire request was cancelled or failed
    nixl_status_t status = sharedState_->status.load();
    if (status == NIXL_SUCCESS) {
        status = nixlUcxBackendH::status();
    }
    return status;
}

/*
 * This class represents a composite request handle for a UCX backend.
 * It is used to encapsulate multiple parallel requests performed by dedicated
 * worker threads of threadpool, with a single request handle, that it returned
 * to the user.
 */
class nixlUcxCompositeBackendH : public nixlUcxBackendH {
public:
    nixlUcxCompositeBackendH(nixlUcxWorker *worker,
                             size_t worker_id,
                             size_t chunk_size,
                             size_t num_chunks)
        : nixlUcxBackendH(worker, worker_id),
          sharedState_(std::make_shared<nixlUcxBackendSharedState>()),
          chunkSize_(chunk_size) {
        sharedState_->chunks.resize(num_chunks);
    }

    size_t
    getChunkSize() const {
        return chunkSize_;
    }

    size_t
    getNumChunks() const {
        return sharedState_ ? sharedState_->chunks.size() : 0;
    }

    void
    startXfer() {
        NIXL_ASSERT(sharedState_->pendingReqs.load() == 0);
        sharedState_->status.store(NIXL_SUCCESS);
        sharedState_->pendingReqs.store(getNumChunks());
    }

    nixlUcxChunkBackendH *
    startChunk(size_t idx, nixlUcxWorker *worker, size_t worker_id) {
        nixlUcxChunkBackendH *chunk = &sharedState_->chunks[idx];
        chunk->startXfer(sharedState_, worker, worker_id);
        return chunk;
    }

    bool
    isComposite() const override {
        return true;
    }

    nixl_status_t
    release() override {
        NIXL_TRACE << *this << " releasing";
        nixl_status_t status = nixlUcxBackendH::release();
        if (sharedState_) {
            // Set failed status to stop progress chunks
            sharedState_->status.store(NIXL_ERR_NOT_FOUND);
            // Reset shared state - it will be effectively released when the last chunk
            // resets the shared state pointer
            sharedState_.reset();
        }

        return status;
    }

    nixl_status_t
    status() override {
        while (getWorker()->progress())
            ;

        if (sharedState_->pendingReqs.load()) {
            return NIXL_IN_PROG;
        }

        nixl_status_t status = nixlUcxBackendH::status();
        if (status != NIXL_SUCCESS) {
            return status;
        }

        return sharedState_->status.load();
    }

    friend std::ostream &
    operator<<(std::ostream &os, const nixlUcxCompositeBackendH &handle) {
        os << "composite handle " << &handle << "{chunks: " << handle.getNumChunks();
        if (handle.sharedState_) {
            os << ", " << *handle.sharedState_;
        } else {
            os << ", state: nullptr";
        }
        return os << "}}";
    }

private:
    std::shared_ptr<nixlUcxBackendSharedState> sharedState_;
    size_t chunkSize_;
};

class nixlUcxDedicatedThread : public nixlUcxThread {
public:
    nixlUcxDedicatedThread(nixlUcxEngine *engine, asio::io_context &io)
        : nixlUcxThread(engine, 1),
          io_(io) {}

    static nixlUcxDedicatedThread *
    getDedicatedThread() {
        return (nixlUcxDedicatedThread *)tlsThread();
    }

    void
    addRequest(nixlUcxChunkBackendH *handle) {
        requests_.push_back(handle);
    }

protected:
    void
    run() override {
        auto guard = asio::make_work_guard(io_);
        NIXL_DEBUG << "dedicated " << *this << " running";

        while (!io_.stopped()) {
            if (!requests_.empty()) {
                io_.poll_one();
            } else {
                NIXL_TRACE << "dedicated " << *this << " waiting for requests";
                io_.run_one();
            }

            if (requests_.empty()) {
                continue;
            }

            for (auto it = requests_.begin(); it != requests_.end();) {
                nixl_status_t status = (*it)->status();
                if (status != NIXL_IN_PROG) {
                    NIXL_TRACE << "dedicated " << *this << " completing " << *(*it)
                               << " with status: " << status;
                    (*it)->complete(status);
                    it = requests_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (!requests_.empty()) {
            NIXL_WARN << "dedicated " << *this << " dropping " << requests_.size()
                      << " requests on exit";
            for (auto it = requests_.begin(); it != requests_.end();) {
                NIXL_INFO << "dropping " << *(*it);
                (*it)->complete(NIXL_ERR_BACKEND);
            }
            requests_.clear();
        }

        NIXL_DEBUG << "dedicated " << *this << " exiting";
    }

private:
    asio::io_context &io_;
    std::vector<nixlUcxChunkBackendH *> requests_;
};

nixlUcxThreadPoolEngine::nixlUcxThreadPoolEngine(const nixlBackendInitParams &init_params)
    : nixlUcxEngine(init_params) {
    size_t num_threads = nixl_b_params_get(init_params.customParams, "num_threads", 0);
    numSharedWorkers_ = getWorkers().size() - num_threads;
    NIXL_ASSERT(numSharedWorkers_ > 0);

    splitBatchSize_ = nixl_b_params_get(init_params.customParams, "split_batch_size", 1024);

    if (init_params.enableProgTh) {
        sharedThread_ =
            std::make_unique<nixlUcxSharedThread>(this, numSharedWorkers_, init_params.pthrDelay);
        for (size_t i = 0; i < numSharedWorkers_; i++) {
            sharedThread_->addWorker(getWorkers()[i].get(), i);
        }
        sharedThread_->start();
    }

    if (num_threads > 0) {
        io_.reset(new asio::io_context());
        dedicatedThreads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            size_t worker_id = numSharedWorkers_ + i;
            dedicatedThreads_.emplace_back(std::make_unique<nixlUcxDedicatedThread>(this, *io_));
            dedicatedThreads_.back()->addWorker(getWorker(worker_id).get(), worker_id);
            dedicatedThreads_.back()->start();
        }
    }
}

nixlUcxThreadPoolEngine::~nixlUcxThreadPoolEngine() {
    if (sharedThread_) {
        sharedThread_->join();
    }

    if (io_) {
        io_->stop();
        for (auto &thread : dedicatedThreads_) {
            thread->join();
        }
    }
}

nixl_status_t
nixlUcxThreadPoolEngine::prepXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH *&handle,
                                  const nixl_opt_b_args_t *opt_args) const {
    size_t batch_size = local.descCount();
    if (batch_size < splitBatchSize_) {
        return nixlUcxEngine::prepXfer(operation, local, remote, remote_agent, handle, opt_args);
    }

    size_t chunk_size = std::max(batch_size / dedicatedThreads_.size(), splitBatchSize_);
    size_t num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    size_t worker_id = getWorkerId();
    auto comp_handle =
        new nixlUcxCompositeBackendH(getWorker(worker_id).get(), worker_id, chunk_size, num_chunks);
    NIXL_TRACE << "created " << *comp_handle;
    handle = comp_handle;
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxThreadPoolEngine::sendXferRange(const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH *handle,
                                       size_t start_idx,
                                       size_t end_idx) const {
    nixlUcxBackendH *int_handle = static_cast<nixlUcxBackendH *>(handle);
    if (!int_handle->isComposite()) {
        return nixlUcxEngine::sendXferRange(
            operation, local, remote, remote_agent, handle, start_idx, end_idx);
    }

    nixlUcxCompositeBackendH *comp_handle = static_cast<nixlUcxCompositeBackendH *>(int_handle);
    comp_handle->startXfer();
    size_t chunk_size = comp_handle->getChunkSize();
    NIXL_TRACE << "sending " << *comp_handle;

    std::promise<void> promise;
    std::future<void> future = promise.get_future();
    std::atomic<size_t> remaining{comp_handle->getNumChunks()};
    std::atomic<nixl_status_t> status{NIXL_SUCCESS};

    for (size_t i = 0; i < comp_handle->getNumChunks(); i++) {
        io_->post([&, i]() {
            auto thread = nixlUcxDedicatedThread::getDedicatedThread();
            NIXL_ASSERT(thread != nullptr);

            nixlUcxChunkBackendH *chunk_handle =
                comp_handle->startChunk(i, thread->getWorkers()[0], thread->getWorkerId());
            NIXL_TRACE << "dedicated " << *thread << " starting " << *chunk_handle;

            size_t start_idx = i * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, (size_t)local.descCount());
            nixl_status_t ret = nixlUcxEngine::sendXferRange(
                operation, local, remote, remote_agent, chunk_handle, start_idx, end_idx);
            if (ret != NIXL_SUCCESS) {
                status.store(ret);
                chunk_handle->complete(ret);
            } else {
                NIXL_TRACE << "dedicated " << *thread << " sent " << *chunk_handle;
                thread->addRequest(chunk_handle);
            }

            if (remaining.fetch_sub(1) == 1) {
                promise.set_value();
            }
        });
    }

    future.wait();
    NIXL_TRACE << "sent " << *comp_handle << " with status: " << status.load();
    return status.load();
}

void
nixlUcxThreadPoolEngine::appendNotif(std::string remote_name, std::string msg) {
    if (nixlUcxThread::isProgressThread(this)) {
        std::lock_guard<std::mutex> lock(notifMutex_);
        notifThread_.emplace_back(std::move(remote_name), std::move(msg));
    } else {
        nixlUcxEngine::appendNotif(std::move(remote_name), std::move(msg));
    }
}

nixl_status_t
nixlUcxThreadPoolEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) return NIXL_ERR_INVALID_PARAM;

    if (!sharedThread_) {
        progress();
    }

    getNotifsImpl(notif_list);
    std::lock_guard<std::mutex> lock(notifMutex_);
    moveNotifList(notifThread_, notif_list);
    return NIXL_SUCCESS;
}

/****************************************
 * Constructor/Destructor
 *****************************************/

std::unique_ptr<nixlUcxEngine>
nixlUcxEngine::create(const nixlBackendInitParams &init_params) {
    nixlUcxEngine *engine;
    size_t num_threads = nixl_b_params_get(init_params.customParams, "num_threads", 0);
    if (num_threads > 0) {
        engine = new nixlUcxThreadPoolEngine(init_params);
    } else if (init_params.enableProgTh) {
        engine = new nixlUcxThreadEngine(init_params);
    } else {
        engine = new nixlUcxEngine(init_params);
    }
    return std::unique_ptr<nixlUcxEngine>(engine);
}

nixlUcxEngine::nixlUcxEngine(const nixlBackendInitParams &init_params)
    : nixlBackendEngine(&init_params),
      sharedWorkerIndex_(1),
      progressThreadEnabled_(init_params.enableProgTh) {
    std::vector<std::string> devs; /* Empty vector */
    nixl_b_params_t *custom_params = init_params.customParams;

    if (custom_params->count("device_list")!=0)
        devs = absl::StrSplit((*custom_params)["device_list"], ", ");

    size_t num_workers = nixl_b_params_get(custom_params, "num_workers", 1);
    size_t num_threads = nixl_b_params_get(custom_params, "num_threads", 0);
    size_t num_device_channels = nixl_b_params_get(custom_params, "ucx_num_device_channels", 4);

    if (num_workers <= num_threads) {
        /* There must be at least one shared worker */
        num_workers = num_threads + 1;
    }

    ucp_err_handling_mode_t err_handling_mode;
    const auto err_handling_mode_it =
        custom_params->find(std::string(nixl_ucx_err_handling_param_name));
    if (err_handling_mode_it == custom_params->end()) {
        err_handling_mode = UCP_ERR_HANDLING_MODE_PEER;
    } else {
        err_handling_mode = ucx_err_mode_from_string(err_handling_mode_it->second);
    }

    const auto engine_config_it = custom_params->find("engine_config");
    const auto engine_config =
        (engine_config_it != custom_params->end()) ? engine_config_it->second : "";

    uc = std::make_unique<nixlUcxContext>(devs,
                                          init_params.enableProgTh,
                                          num_workers,
                                          init_params.syncMode,
                                          num_device_channels,
                                          engine_config);

    uc->warnAboutHardwareSupportMismatch();

    for (size_t i = 0; i < num_workers; i++) {
        uws.emplace_back(std::make_unique<nixlUcxWorker>(*uc, err_handling_mode));
    }

    auto &uw = uws.front();
    workerAddr = uw->epAddr();
    uw->regAmCallback(NOTIF_STR, notifAmCb, this);
}

nixl_mem_list_t nixlUcxEngine::getSupportedMems () const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    mems.push_back(VRAM_SEG);
    return mems;
}

static std::unordered_map<const nixlUcxEngine *, size_t> &
tlsSharedWorkerMap() {
    static thread_local std::unordered_map<const nixlUcxEngine *, size_t> map;
    return map;
}

// Through parent destructor the unregister will be called.
nixlUcxEngine::~nixlUcxEngine() {
    tlsSharedWorkerMap().erase(this);
}

/****************************************
 * Connection management
*****************************************/

nixl_status_t nixlUcxEngine::checkConn(const std::string &remote_agent) {
    return remoteConnMap.count(remote_agent) ? NIXL_SUCCESS : NIXL_ERR_NOT_FOUND;
}

nixl_status_t nixlUcxEngine::getConnInfo(std::string &str) const {
    str = workerAddr;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::connect(const std::string &remote_agent) {
    if(remote_agent == localAgent) {
        return loadRemoteConnInfo(remote_agent, workerAddr);
    }

    return (remoteConnMap.find(remote_agent) == remoteConnMap.end()) ? NIXL_ERR_NOT_FOUND :
                                                                       NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::disconnect(const std::string &remote_agent) {
    auto search = remoteConnMap.find(remote_agent);

    if (search == remoteConnMap.end()) {
        return NIXL_ERR_NOT_FOUND;
    }

    // thread safety?
    remoteConnMap.erase(search);
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::loadRemoteConnInfo (const std::string &remote_agent,
                                                 const std::string &remote_conn_info)
{
    size_t size = remote_conn_info.size();
    std::vector<char> addr(size);

    if(remoteConnMap.count(remote_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlSerDes::_stringToBytes(addr.data(), remote_conn_info, size);
    std::shared_ptr<nixlUcxConnection> conn = std::make_shared<nixlUcxConnection>();
    for (auto &uw: uws) {
        auto result = uw->connect(addr.data(), size);
        if (!result.ok()) {
            return NIXL_ERR_BACKEND;
        }
        conn->eps.push_back(std::move(*result));
    }

    remoteConnMap.insert({remote_agent, conn});

    return NIXL_SUCCESS;
}

/****************************************
 * Memory management
*****************************************/
nixl_status_t nixlUcxEngine::registerMem (const nixlBlobDesc &mem,
                                          const nixl_mem_t &nixl_mem,
                                          nixlBackendMD* &out)
{
    auto priv = std::make_unique<nixlUcxPrivateMetadata>();

    // TODO: Add nixl_mem check?
    const int ret = uc->memReg((void*) mem.addr, mem.len, priv->mem, nixl_mem);
    if (ret) {
        return NIXL_ERR_BACKEND;
    }
    priv->rkeyStr = uc->packRkey(priv->mem);

    if (priv->rkeyStr.empty()) {
        return NIXL_ERR_BACKEND;
    }
    out = priv.release();
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::deregisterMem (nixlBackendMD* meta)
{
    nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    uc->memDereg(priv->mem);
    delete priv;
    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::getPublicData (const nixlBackendMD* meta,
                                            std::string &str) const {
    const nixlUcxPrivateMetadata *priv = (nixlUcxPrivateMetadata*) meta;
    str = priv->get();
    return NIXL_SUCCESS;
}


// To be cleaned up
nixl_status_t
nixlUcxEngine::internalMDHelper (const nixl_blob_t &blob,
                                 const std::string &agent,
                                 nixlBackendMD* &output) {
    try {
        auto md = std::make_unique<nixlUcxPublicMetadata>();
        size_t size = blob.size();

        auto search = remoteConnMap.find(agent);

        if (search == remoteConnMap.end()) {
            // TODO: err: remote connection not found
            return NIXL_ERR_NOT_FOUND;
        }
        md->conn = search->second;

        std::vector<char> addr(size);
        nixlSerDes::_stringToBytes(addr.data(), blob, size);

        for (size_t wid = 0; wid < uws.size(); wid++) {
            md->addRkey(*md->conn->getEp(wid), addr.data());
        }

        output = (nixlBackendMD *)md.release();

        return NIXL_SUCCESS;
    }
    catch (const std::runtime_error &e) {
        NIXL_ERROR << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::loadLocalMD (nixlBackendMD* input,
                            nixlBackendMD* &output)
{
    nixlUcxPrivateMetadata* input_md = (nixlUcxPrivateMetadata*) input;
    return internalMDHelper(input_md->rkeyStr, localAgent, output);
}

// To be cleaned up
nixl_status_t nixlUcxEngine::loadRemoteMD (const nixlBlobDesc &input,
                                           const nixl_mem_t &nixl_mem,
                                           const std::string &remote_agent,
                                           nixlBackendMD* &output)
{
    return internalMDHelper(input.metaInfo, remote_agent, output);
}

nixl_status_t nixlUcxEngine::unloadMD (nixlBackendMD* input) {

    nixlUcxPublicMetadata *md = (nixlUcxPublicMetadata*) input; //typecast?
    delete md;

    return NIXL_SUCCESS;
}

/****************************************
 * Data movement
*****************************************/

size_t
nixlUcxEngine::getWorkerId(const nixl_opt_b_args_t *opt_args) const noexcept {
    if (opt_args) {
        const auto worker_id = getWorkerIdFromOptArgs(*opt_args);
        if (worker_id) {
            return *worker_id;
        }
    }

    auto it = tlsSharedWorkerMap().find(this);
    if (it == tlsSharedWorkerMap().end()) {
        size_t index = sharedWorkerIndex_.fetch_add(1) % getSharedWorkersSize();
        it = tlsSharedWorkerMap().emplace(this, index).first;
        NIXL_DEBUG << "engine " << this << " bound shared worker " << index << " to thread "
                   << std::this_thread::get_id();
    }
    return it->second;
}

std::optional<size_t>
nixlUcxEngine::getWorkerIdFromOptArgs(const nixl_opt_b_args_t &opt_args) const noexcept {
    constexpr std::string_view worker_id_key = "worker_id=";
    size_t pos = opt_args.customParam.find(worker_id_key);
    if (pos == std::string::npos) {
        return std::nullopt;
    }

    try {
        size_t worker_id = std::stoull(opt_args.customParam.substr(pos + worker_id_key.length()));

        if (worker_id >= getSharedWorkersSize()) {
            NIXL_WARN << "Invalid worker_id " << worker_id << " (must be < "
                      << getSharedWorkersSize() << ")";
            return std::nullopt;
        }

        return worker_id;
    }
    catch (const std::exception &e) {
        NIXL_WARN << "Failed to parse worker_id from customParam: " << e.what();
        return std::nullopt;
    }
}

nixl_status_t nixlUcxEngine::prepXfer (const nixl_xfer_op_t &operation,
                                       const nixl_meta_dlist_t &local,
                                       const nixl_meta_dlist_t &remote,
                                       const std::string &remote_agent,
                                       nixlBackendReqH* &handle,
                                       const nixl_opt_b_args_t* opt_args) const
{
    if (local.descCount() == 0 || remote.descCount() == 0) {
        NIXL_ERROR << "Local or remote descriptor list is empty";
        return NIXL_ERR_INVALID_PARAM;
    }

    const auto worker_id = getWorkerId(opt_args);
    /* TODO: try to get from a pool first */
    /* When a UCX progress thread is enabled it owns the worker, so the waiting
     * thread must not also drive progress (avoid double-progress); it only
     * reads the completion counters. */
    auto *ucx_handle =
        new nixlUcxBackendH(getWorker(worker_id).get(), worker_id, !progressThreadEnabled_);

    handle = ucx_handle;

    return NIXL_SUCCESS;
}

nixl_status_t nixlUcxEngine::estimateXferCost (const nixl_xfer_op_t &operation,
                                               const nixl_meta_dlist_t &local,
                                               const nixl_meta_dlist_t &remote,
                                               const std::string &remote_agent,
                                               nixlBackendReqH* const &handle,
                                               std::chrono::microseconds &duration,
                                               std::chrono::microseconds &err_margin,
                                               nixl_cost_t &method,
                                               const nixl_opt_args_t* opt_args) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    size_t workerId = intHandle->getWorkerId();

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "Local (" << local.descCount() << ") and remote (" << remote.descCount()
                   << ") descriptor lists differ in size for cost estimation";
        return NIXL_ERR_MISMATCH;
    }

    duration = std::chrono::microseconds(0);
    err_margin = std::chrono::microseconds(0);

    if (local.descCount() == 0) {
        // Nothing to do, use a default value
        method = nixl_cost_t::ANALYTICAL_BACKEND;
        return NIXL_SUCCESS;
    }

    for (int i = 0; i < local.descCount(); i++) {
        size_t lsize = local[i].len;
        size_t rsize = remote[i].len;

        nixlUcxPrivateMetadata *lmd = static_cast<nixlUcxPrivateMetadata*>(local[i].metadataP);
        nixlUcxPublicMetadata *rmd = static_cast<nixlUcxPublicMetadata*>(remote[i].metadataP);

        NIXL_ASSERT(lmd && rmd) << "No metadata found in descriptor lists at index " << i << " during cost estimation";
        NIXL_ASSERT(lsize == rsize) << "Local size (" << lsize << ") != Remote size (" << rsize
                                    << ") at index " << i << " during cost estimation";

        std::chrono::microseconds msg_duration;
        std::chrono::microseconds msg_err_margin;
        nixl_cost_t msg_method;
        nixl_status_t ret = rmd->conn->getEp(workerId)->estimateCost(lsize, msg_duration, msg_err_margin, msg_method);
        if (ret != NIXL_SUCCESS) {
            NIXL_ERROR << "Worker failed to estimate cost for segment " << i << " status: " << ret;
            return ret;
        }

        duration += msg_duration;
        err_margin += msg_err_margin;
        method = msg_method;
    }

    return NIXL_SUCCESS;
}

nixlUcxEngine::batchResult
nixlUcxEngine::sendXferRangeBatch(nixlUcxBackendH *handle,
                                  ucx_connection_ptr_t conn,
                                  nixlUcxEp &ep,
                                  nixl_xfer_op_t operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  size_t worker_id,
                                  size_t start_idx,
                                  size_t end_idx) {
    batchResult result = {NIXL_SUCCESS, 0};
    nixlUcxCompletionState *completion = handle->completionState();

    /* Record the connection once for the whole EP batch (used by the flush
     * phase), rather than once per request. */
    handle->addConnection(conn);

    for (size_t i = start_idx; i < end_idx; ++i) {
        void *laddr = (void *)local[i].addr;
        size_t lsize = local[i].len;
        uint64_t raddr = static_cast<uint64_t>(remote[i].addr);
        NIXL_ASSERT(lsize == remote[i].len);

        auto lmd = static_cast<nixlUcxPrivateMetadata *>(local[i].metadataP);
        auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);
        auto &rmd_ep = rmd->conn->getEp(worker_id);
        if (__builtin_expect(rmd_ep.get() != &ep, 0)) {
            break;
        }

        ++result.size;
        nixlUcxReq req;
        /* Attach a completion callback to every op. The callback bumps the
         * completion counter and frees its own request, so the request is never
         * stored here and never freed before its callback fires.
         *
         * Register the expected callback *before* submitting, so a callback
         * firing on the progress thread is always accounted for first. */
        completion->expect();
        nixl_status_t ret = operation == NIXL_READ ?
            ep.read(raddr,
                    rmd->getRkey(worker_id),
                    laddr,
                    lmd->mem,
                    lsize,
                    req,
                    nixlUcxXferCompletionCb,
                    completion) :
            ep.write(laddr,
                     lmd->mem,
                     raddr,
                     rmd->getRkey(worker_id),
                     lsize,
                     req,
                     nixlUcxXferCompletionCb,
                     completion);

        nixl_status_t append_ret = handle->append(ret);
        if (append_ret != NIXL_SUCCESS) {
            result.status = append_ret;
            break;
        }
    }

    return result;
}

nixl_status_t
nixlUcxEngine::sendXferRange(const nixl_xfer_op_t &operation,
                             const nixl_meta_dlist_t &local,
                             const nixl_meta_dlist_t &remote,
                             const std::string &remote_agent,
                             nixlBackendReqH *handle,
                             size_t start_idx,
                             size_t end_idx) const {
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    size_t workerId = intHandle->getWorkerId();
    nixl_status_t ret;

    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        return NIXL_ERR_INVALID_PARAM;
    }

    /* RMA and flush requests are callback-tracked (freed by their callbacks),
     * so requests_ only ever holds the optional notification request. */
    intHandle->reserve(2);

    for (size_t i = start_idx; i < end_idx;) {
        /* Send requests to a single EP */
        auto rmd = static_cast<nixlUcxPublicMetadata *>(remote[i].metadataP);
        auto &ep = rmd->conn->getEp(workerId);
        /* Submits and appends every in-progress request for this EP batch */
        auto result = sendXferRangeBatch(
            intHandle, rmd->conn, *ep, operation, local, remote, workerId, i, end_idx);
        if (result.status != NIXL_SUCCESS) {
            return result.status;
        }

        i += result.size;
    }

    /*
     * Flush keeps intHandle non-empty until the operation is actually
     * completed, which can happen after local requests completion.
     * We need to flush all distinct connections to ensure that the operation
     * is actually completed. The flush request also carries a completion
     * callback so its completion is observed via the same atomic counter.
     */
    for (auto &conn : intHandle->getConnections()) {
        nixlUcxReq req;
        /* Register the expected flush callback before submitting it. */
        intHandle->completionState()->expect();
        ret = conn->getEp(workerId)->flushEp(req, nixlUcxXferCompletionCb, intHandle->completionState());
        if (intHandle->append(ret) != NIXL_SUCCESS) {
            return ret;
        }
    }

    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::postXfer(const nixl_xfer_op_t &operation,
                        const nixl_meta_dlist_t &local,
                        const nixl_meta_dlist_t &remote,
                        const std::string &remote_agent,
                        nixlBackendReqH *&handle,
                        const nixl_opt_b_args_t *opt_args) const {
    size_t lcnt = local.descCount();
    size_t rcnt = remote.descCount();
    nixlUcxBackendH *int_handle = static_cast<nixlUcxBackendH *>(handle);
    nixl_status_t ret;

    if (lcnt != rcnt) {
        NIXL_ERROR << "Local (" << lcnt << ") and remote (" << rcnt
                   << ") descriptor lists differ in size";
        return NIXL_ERR_INVALID_PARAM;
    }

    // TODO: assert that handle is empty/completed, as we can't post request before completion

    ret = sendXferRange(operation, local, remote, remote_agent, handle, 0, lcnt);
    if (ret != NIXL_SUCCESS) {
        return ret;
    }

    ret = int_handle->status();
    if (opt_args && opt_args->hasNotif) {
        if (ret == NIXL_SUCCESS) {
            nixlUcxReq req;
            auto rmd = (nixlUcxPublicMetadata *)remote[0].metadataP;
            ret = notifSendPriv(remote_agent,
                                opt_args->notifMsg,
                                rmd->conn->getEp(int_handle->getWorkerId()),
                                &req);
            if (int_handle->appendLegacy(ret, req, rmd->conn) != NIXL_SUCCESS) {
                return ret;
            }

            ret = int_handle->status();
        } else if (ret == NIXL_IN_PROG) {
            int_handle->notification().emplace(remote_agent, opt_args->notifMsg);
        }
    }

    return ret;
}

nixl_status_t nixlUcxEngine::checkXfer (nixlBackendReqH* handle) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    auto& notif = intHandle->notification();
    nixl_status_t handle_status = intHandle->status();

    if ((handle_status != NIXL_SUCCESS) || !notif.has_value()) {
        if (handle_status != NIXL_IN_PROG) { // error flow
            notif.reset();
        }

        return handle_status;
    }

    ucx_connection_ptr_t conn = getConnection(notif->agent);
    if (!conn) {
        notif.reset();
        return NIXL_ERR_NOT_FOUND;
    }

    nixlUcxReq req;
    nixl_status_t status = notifSendPriv(notif->agent,
                                         notif->payload,
                                         conn->getEp(intHandle->getWorkerId()),
                                         &req);
    notif.reset();

    if (intHandle->appendLegacy(status, req, conn) != NIXL_SUCCESS) {
        return status;
    }

    return intHandle->status();
}

nixl_status_t nixlUcxEngine::releaseReqH(nixlBackendReqH* handle) const
{
    nixlUcxBackendH *intHandle = (nixlUcxBackendH *)handle;
    nixl_status_t status = intHandle->release();

    /* TODO: return to a pool instead. */
    delete intHandle;

    return status;
}

int nixlUcxEngine::progress() {
    // TODO: add listen for connection handling if necessary
    int ret = 0;
    for (auto &uw: uws)
        ret += uw->progress();
    return ret;
}

/****************************************
 * Notifications
*****************************************/

//agent will provide cached msg
nixl_status_t
nixlUcxEngine::notifSendPriv(const std::string &remote_agent,
                             const std::string &msg,
                             const std::unique_ptr<nixlUcxEp> &ep,
                             nixlUcxReq *req) const {
    nixlSerDes ser_des;

    ser_des.addStr("name", localAgent);
    ser_des.addStr("msg", msg);
    // TODO: replace with mpool for performance

    std::string *buffer = new std::string(ser_des.exportStr());
    auto deleter = [buffer, req](void *completed_request, void *ptr) {
        delete buffer;
        if ((req == nullptr) && (completed_request != nullptr)) {
            /* Caller is not interested in the request, free it */
            ucp_request_free(completed_request);
        }
    };

    return ep->sendAm(NOTIF_STR,
                      nullptr,
                      0,
                      (void *)buffer->data(),
                      buffer->size(),
                      UCP_AM_SEND_FLAG_EAGER,
                      req,
                      deleter);
}

ucx_connection_ptr_t
nixlUcxEngine::getConnection(const std::string &remote_agent) const {
    auto search = remoteConnMap.find(remote_agent);
    return (search != remoteConnMap.end()) ? search->second : nullptr;
}

void
nixlUcxEngine::appendNotif(std::string remote_name, std::string msg) {
    notifMainList.emplace_back(std::move(remote_name), std::move(msg));
}

ucs_status_t
nixlUcxEngine::notifAmCb(void *arg, const void *header,
                         size_t header_length, void *data,
                         size_t length,
                         const ucp_am_recv_param_t *param)
{
    nixlSerDes ser_des;

    std::string ser_str( (char*) data, length);
    nixlUcxEngine* engine = (nixlUcxEngine*) arg;

    // send_am should be forcing EAGER protocol
    NIXL_ASSERT(!(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV));
    NIXL_ASSERT(header_length == 0) << "header_length " << header_length;

    ser_des.importStr(ser_str);
    std::string remote_name = ser_des.getStr("name");
    std::string msg = ser_des.getStr("msg");

    engine->appendNotif(std::move(remote_name), std::move(msg));
    return UCS_OK;
}

void
nixlUcxEngine::getNotifsImpl(notif_list_t &notif_list) {
    moveNotifList(notifMainList, notif_list);
}

nixl_status_t nixlUcxEngine::getNotifs(notif_list_t &notif_list)
{
    if (!notif_list.empty()) return NIXL_ERR_INVALID_PARAM;

    while (progress())
        ;
    getNotifsImpl(notif_list);
    return NIXL_SUCCESS;
}

nixl_status_t
nixlUcxEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    auto conn = getConnection(remote_agent);
    if (!conn) {
        return NIXL_ERR_NOT_FOUND;
    }

    nixl_status_t ret = notifSendPriv(remote_agent, msg, conn->getEp(getWorkerId()));
    if (ret == NIXL_IN_PROG) {
        ret = NIXL_SUCCESS;
    }
    return ret;
}

nixl_status_t
nixlUcxEngine::prepMemView(const nixl_remote_meta_dlist_t &dlist,
                           nixlMemViewH &mvh,
                           const nixl_opt_b_args_t *opt_args) const {
    const size_t worker_id = getWorkerId(opt_args);
    try {
        mvh = nixl::ucx::createMemList(dlist, worker_id, *getWorker(worker_id));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to prepare remote memory view: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

nixl_status_t
nixlUcxEngine::prepMemView(const nixl_meta_dlist_t &dlist,
                           nixlMemViewH &mvh,
                           const nixl_opt_b_args_t *opt_args) const {
    const size_t worker_id = getWorkerId(opt_args);
    try {
        mvh = nixl::ucx::createMemList(dlist, *getWorker(worker_id));
        return NIXL_SUCCESS;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Failed to prepare local memory view: " << e.what();
        return NIXL_ERR_BACKEND;
    }
}

void
nixlUcxEngine::releaseMemView(nixlMemViewH mem_view) const {
    nixl::ucx::releaseMemList(mem_view);
}
