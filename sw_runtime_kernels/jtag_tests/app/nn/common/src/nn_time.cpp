/*
* {% copyright %}
*/
#include "nn_time.h"
#include "registersMyriad.h"
#include "DrvRegUtilsDefines.h"

using namespace std;

namespace
{
    uint64_t get_ticks()
    {
        uint32_t base_address = TIM2_BASE_ADR;
        uint32_t low = GET_REG_WORD_VAL(base_address + TIM_FREE_LOWER_RAW_OFFSET);
        uint32_t high = GET_REG_WORD_VAL(base_address + TIM_FREE_CNT1_OFFSET);

        // Check if timer overflowed, and adjust
        uint32_t updated = GET_REG_WORD_VAL(TIM2_FREE_LOWER_RAW_ADR);

        if(updated < low)
            --high;

        return ((uint64_t)high << 32) | ((uint64_t)low);
    }
}

namespace nn
{
    namespace time
    {
        Timer::Timer() :
            start_(Clock::now())
        {
        }

        Timer::~Timer()
        {
        }

        void Timer::start()
        {
            start_ = Clock::now();
        }

        size_t Timer::elapsedMs() const
        {
            auto now = Clock::now();
            return static_cast<size_t>(chrono::duration_cast<chrono::milliseconds>(now - start_).count());
        }

        size_t Timer::elapsedUs() const
        {
            auto now = Clock::now();
            return static_cast<size_t>(chrono::duration_cast<chrono::microseconds>(now - start_).count());
        }

        size_t Timer::elapsedNs() const
        {
            auto now = Clock::now();
            return static_cast<size_t>(chrono::duration_cast<chrono::nanoseconds>(now - start_).count());
        }

        Ticker::Ticker() :
            start_(0)
        {
            start();
        }

        Ticker::~Ticker()
        {
        }

        void Ticker::start()
        {
            start_ = get_ticks();
        }

        unsigned long long Ticker::ticks() const
        {
            unsigned long long now = get_ticks();
            return now - start_;
        }
    }
}
