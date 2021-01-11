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
#include <mutex>
#include <thread>

namespace vpux {

class WatchDog {
private:
    volatile bool _stopWatchdog = false;
    volatile bool _pauseWatchdog = true;
    std::chrono::steady_clock::time_point _start;

    std::thread _abortThread;
    std::condition_variable _cvPeriodic;
    std::function<void()> _abortProgram;
    vpu::Logger::Ptr _logger;

public:
    WatchDog(const uint32_t watchdog_milliseconds,
             vpu::Logger::Ptr logger,
             std::function<void()> abort_callback)  : _abortProgram(abort_callback), _logger(logger) {

        if (watchdog_milliseconds == 0) {
            _logger->warning("[WATCHDOG] disabled");
            return;
        }
        _logger->info("[WATCHDOG] Starting with timeout %d ms", watchdog_milliseconds);
        // Start a thread which will abort our program after a time.
        _abortThread = std::thread(&WatchDog::watchdog_thread, this, watchdog_milliseconds);
    }

    ~WatchDog() {
        _logger->info("[WATCHDOG] stop initiated");
        auto wd_stop_begins = std::chrono::steady_clock::now();
        _stopWatchdog = true;
        if (_abortThread.joinable()) {
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

    // Reset our timeout.
    void Start(void) {
        _start = std::chrono::steady_clock::now();
        _pauseWatchdog = false;
    }

    void Pause() { _pauseWatchdog = true; }

private:
    // If we don't get kicked then we will abort the program.
    void watchdog_thread(const uint32_t timeout_ms) {
        using namespace std::chrono;
        _start = steady_clock::now();

        std::mutex mtx;
        std::unique_lock<std::mutex> lck(mtx);

        while (duration_cast<milliseconds>(steady_clock::now() - _start).count() < timeout_ms && !_stopWatchdog) {
            _cvPeriodic.wait_for(lck, std::chrono::seconds(1));
            if (_pauseWatchdog) {
                _start = steady_clock::now();
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