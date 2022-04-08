/*
* {% copyright %}
*/
#ifndef NN_TIME_H_
#define NN_TIME_H_

#include <chrono>

namespace nn
{
    namespace time
    {
        class Timer
        {
            typedef std::chrono::high_resolution_clock Clock;

        public:
            Timer();
            ~Timer();

            void start();
            size_t elapsedMs() const;
            size_t elapsedUs() const;
            size_t elapsedNs() const;

        private:
            Clock::time_point start_;
        };

        class Ticker
        {
        public:
            Ticker();
            ~Ticker();

            void start();
            unsigned long long ticks() const;

        private:
            unsigned long long start_;
        };
    }
}

#endif // NN_TIME_H_
