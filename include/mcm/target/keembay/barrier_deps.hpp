#ifndef BARRIER_DEPS_HPP
#define BARRIER_DEPS_HPP

#include <vector>
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{
    class BarrierDependencies : public LogSender
    {
        int waitBarrier_;
        std::vector<unsigned> updateBarriers_;
    
    public:
        BarrierDependencies();
        void setWaitBarrier(int barrierId);
        void addUpdateBarrier(int barrierId);

        int getWait();
        std::vector<unsigned> getUpdate();

        std::string getLogID() const override;
        std::string toString() const;
    };
}

#endif // BARRIER_DEPS_HPP
