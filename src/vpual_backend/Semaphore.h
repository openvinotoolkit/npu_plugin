//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <mutex>
#include <condition_variable>

class Semaphore {
public:
    Semaphore(int count_ = 0): count(count_) {}

    inline void notify()
    {
        std::unique_lock<std::mutex> lock(mtx);
        count++;
        // Notify the waiting thread
        cv.notify_one();
    }
    inline void wait()
    {
        std::unique_lock<std::mutex> lock(mtx);
        while(count == 0){
            // Wait on the mutex until notify is called
            cv.wait(lock);
        }
        count--;
    }
    inline void count_one()
    {
        count = 1;
    }
    inline void count_zero()
    {
        count = 0;
    }
private:
    std::mutex mtx;
    std::condition_variable cv;
    volatile int count;
};
