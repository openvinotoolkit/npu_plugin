//
// Copyright 2021 Intel Corporation.
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
