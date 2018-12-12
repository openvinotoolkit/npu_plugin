#ifndef MV_RUNTIME_MODEL_BARRIER_
#define MV_RUNTIME_MODEL_BARRIER_

#include <vector>

namespace mv
{
    struct RuntimeModelBarrier
    {
        unsigned barrierID;
        unsigned consumerCount;
        unsigned producerCount;
    };

    struct RuntimeModelBarrierReference
    {
        unsigned waitBarrier;
        std::vector<unsigned> updateBarriers;
    };
}

#endif
