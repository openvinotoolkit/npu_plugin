#ifndef BARRIER_DEPS_HPP
#define BARRIER_DEPS_HPP

#include <vector>
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{
    class BarrierDependencies : public LogSender
    {
        //TODO(vamsikku): why is barrierID int and these are unsigned ?
        std::vector<unsigned> waitBarriers_;
        std::vector<unsigned> updateBarriers_;

        void toStringBarrierVector(const std::vector<unsigned>&,
              std::string& ) const;
    
    public:

        BarrierDependencies();
        bool hasWaitBarriers() const { return !(waitBarriers_.empty()); }
        void addUpdateBarrier(int barrierId);
        void addWaitBarrier(int barrierId);

        void clear() { waitBarriers_.clear(); updateBarriers_.clear(); }

        const std::vector<unsigned>& getWait();
        const std::vector<unsigned>& getUpdate();

        bool hasWaitBarrierWithID(unsigned id) const {
          for (auto itr=waitBarriers_.begin();
                itr!=waitBarriers_.end(); ++itr) {
            if (*itr == id) { return true; }
          }
          return false;
        }

        bool hasUpdateBarrierWithID(unsigned id) const {
          for (auto itr=updateBarriers_.begin();
                itr!=updateBarriers_.end(); ++itr) {
            if (*itr == id) { return true; }
          }
          return false;
        }

        std::string getLogID() const override;
        std::string toString() const;
    };
}

#endif // BARRIER_DEPS_HPP
