// {% copyright %}
#ifndef MV_TENSOR_TIMER_H_
#define MV_TENSOR_TIMER_H_

#include <mv_types.h>
#include <chrono>

namespace mv
{
    namespace tensor
    {
        class Timer
        {
            typedef std::chrono::high_resolution_clock Clock;
        public:
            explicit Timer(double *store = nullptr);
            ~Timer();

            void reset();
            double elapsed() const;

        private:
            Clock::time_point start_;
            double *store_;
            //u64 timestamp_;
            //u32 ticksPerUs_;
        };
    }
}

#endif // MV_TENSOR_TIMER_H_
