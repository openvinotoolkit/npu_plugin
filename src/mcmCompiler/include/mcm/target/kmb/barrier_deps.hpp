#ifndef BARRIER_DEPS_HPP
#define BARRIER_DEPS_HPP

#include <vector>
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{
    class BarrierDependencies : public LogSender
    {
    public:
        BarrierDependencies() = default;
        void addUpdateBarrier(uint32_t id);
        void addWaitBarrier(uint32_t id);
        void removeUpdateBarrier(uint32_t id);
        void removeWaitBarrier(uint32_t id);
        bool hasWaitBarriers() const;
        bool hasUpdateBarrierWithID(uint32_t) const;
        bool hasWaitBarrierWithID(uint32_t) const;
        void clear();
        void clearUpdateBarriers(void);
        const std::vector<uint32_t>& getWait() const;
        const std::vector<uint32_t>& getUpdate() const;
        std::size_t getUpdateSize() const;
        std::size_t getWaitSize() const;
        std::string getLogID() const override;
        std::string toString() const;
    private:
        std::vector<uint32_t> waitBarriers_;
        std::vector<uint32_t> updateBarriers_;
    };
}

#endif // BARRIER_DEPS_HPP
