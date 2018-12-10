#ifndef MV_RUNTIME_MODEL_BARRIER_
#define MV_RUNTIME_MODEL_BARRIER_

#include <vector>

namespace mv
{
    struct RuntimeModelBarrier
    {
        unsigned barrierID_;
        unsigned consumerCount_;
        unsigned producerCount_;
    };

    struct RuntimeModelBarrierReference
    {
        unsigned waitBarrier_;
        std::vector<unsigned> updateBarriers_;
    };
}

#endif
