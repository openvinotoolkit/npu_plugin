//
// Copyright 2019-2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <stdint.h>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vpu/utils/logger.hpp>

namespace vpux {

class WatchDog {
protected:
    volatile bool _stopWatchdog = false;

    std::unordered_map<uintptr_t, std::chrono::steady_clock::time_point> _watchers;
    std::thread _abortThread;
    std::mutex watchersAccess;
    std::condition_variable _cvPeriodic;
    std::chrono::milliseconds _periodicInterval;
    std::function<void()> _abortProgram;
    vpu::Logger::Ptr _logger;

public:
    WatchDog(const uint32_t watchdog_milliseconds,
             vpu::Logger::Ptr logger,
             std::function<void()> abort_callback,
             std::chrono::milliseconds periodic_interval_ms = std::chrono::milliseconds (1000))
            : _periodicInterval(periodic_interval_ms), _abortProgram(abort_callback), _logger(logger) {

        if (watchdog_milliseconds == 0) {
            _logger->warning("[WATCHDOG] disabled");
            return;
        }
        _logger->info("[WATCHDOG] Starting with timeout %d ms", watchdog_milliseconds);
        // Start a thread which will abort our program after a time.
        _abortThread = std::thread(&WatchDog::watchdog_thread, this, watchdog_milliseconds);
    }

    virtual ~WatchDog() {
        _logger->info("[WATCHDOG] stop initiated");
        auto wd_stop_begins = std::chrono::steady_clock::now();

        std::unique_lock<std::mutex> lck(watchersAccess);
        _stopWatchdog = true;
        if (_abortThread.joinable()) {
            lck.unlock();
            _cvPeriodic.notify_one();
            _abortThread.join();
        }
        auto wd_stopped_in = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - wd_stop_begins).count();
        _logger->info("[WATCHDOG] stop completed in %f ms", wd_stopped_in);
    }

    WatchDog(const WatchDog&) = delete;
    WatchDog& operator=(const WatchDog&) = delete;

    WatchDog(WatchDog&&) = delete;
    WatchDog& operator=(WatchDog&&) = delete;

    // start watching our timeout
    void Start() {
        StartImpl(0);
    }
    template <class Id>
    typename std::enable_if<std::is_pointer<Id>::value, void>::type
    Start(Id id) {
        StartImpl(reinterpret_cast<uintptr_t>(id));
    }

    // Reset our timeout
    void Pause() {
        PauseImpl(0);
    }
    template <class Id>
    typename std::enable_if<std::is_pointer<Id>::value, void>::type
    Pause(Id id) {
        PauseImpl(reinterpret_cast<uintptr_t>(id));
    }

private:
    void StartImpl(uintptr_t id) {
        std::unique_lock<std::mutex> lk(watchersAccess);

        auto & watcher = _watchers[id];
        if (watcher.time_since_epoch() != std::chrono::steady_clock::duration::zero()) {
            throw std::runtime_error("Watchdog unexpected Start() for id=" + std::to_string(id));
        }
        watcher = std::chrono::steady_clock::now();
    }
    void PauseImpl(uintptr_t id) {
        std::unique_lock<std::mutex> lk(watchersAccess);
        auto watcher = _watchers.find(id);
        if (watcher == _watchers.end() ||
            watcher->second.time_since_epoch() == std::chrono::steady_clock::duration::zero()) {
            throw std::runtime_error("Watchdog unexpected Pause() for id=" + std::to_string(id));
        }
        watcher->second = std::chrono::steady_clock::time_point{};
    }

    // If we don't get kicked then we will abort the program.
    void watchdog_thread(const uint32_t timeout_ms) {
        using namespace std::chrono;

        std::unique_lock<std::mutex> lck(watchersAccess);
        bool timeout = false;
        while (!_stopWatchdog && !timeout) {
            _cvPeriodic.wait_for(lck, _periodicInterval);
            for (auto && watcher : _watchers) {
                auto startPoint = watcher.second;
                if (startPoint.time_since_epoch() == steady_clock::duration::zero()) {
                    continue;
                }
                if (duration_cast<milliseconds>(steady_clock::now() - startPoint).count() < timeout_ms) {
                    continue;
                }
                timeout = true;
                break;
            }
        }
        if (_stopWatchdog) {
            _logger->info("[WATCHDOG] thread exited");
            return;
        }
        _logger->warning("[WATCHDOG] triggered timeout of %d ms" , timeout_ms);

        _abortProgram();
    }
};

}  // namespace vpux