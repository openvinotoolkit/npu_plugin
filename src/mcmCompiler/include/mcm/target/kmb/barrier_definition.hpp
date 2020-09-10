#ifndef BARRIER_DEFINITION_HPP
#define BARRIER_DEFINITION_HPP

#include <string>
#include <set>

#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    class Barrier : public LogSender
    {
        int group_;
        int index_;
        int barrierID_;
        int realBarrierIndex_;
        int numProducers_;
        int numConsumers_;
        std::set<std::string> producers_; // names of the input ops
        std::set<std::string> consumers_; // names of the op that consume this barrier
        static int barrierCounter_;

    public:
        Barrier(int group = -1, int index = -1);
        Barrier(int group, int index, std::set<std::string>& producers, std::set<std::string>& consumers);
        Barrier(std::set<std::string>& producers, std::set<std::string>& consumers);
        bool operator==(const Barrier& other);
        bool operator!=(const Barrier& other);

        int getGroup() const;
        int getIndex() const;

        int getID() const;
        void setID(int id);

        void setRealBarrierIndex(int real_barrier_index) {
          realBarrierIndex_ = real_barrier_index;
        }
        int getRealBarrierIndex(void) const { return realBarrierIndex_; }

        int getNumProducers() const;
        int getNumConsumers() const;

        /**
         * Note#1: These APIs are required to support workloads in DPU tasks. If these APIs are
         * invoked, numProducers will no longer be the same size as producers_.size(). Same
         * comment for numConsumers as well.
         * Note#2: numProducers and numConsumers does not affect interference graph generation or
         * it's coloring.
         */
        void setNumProducers(int producers);
        void setNumConsumers(int consumers);

        void setGroup(int group);
        void setIndex(int index); // sets HW index for this barrier. Guaranteed to be in the 0-7 range.

        void addProducer(const std::string& producer);
        void addConsumer(const std::string& consumer);
        void removeProducer(const std::string& producer);
        void removeConsumer(const std::string& consumer);

        void clear();
        //TODO(vamsikku): consolidate clear() and this method after the new 
        //barrier scheduler is set as default//
        void clearProducersConsumers() {
          numProducers_ = 0;
          numConsumers_ = 0;
          producers_.clear();
          consumers_.clear();
        }

        bool hasProducers() const;
        bool hasConsumers() const;

        std::set<std::string> getProducers() const;
        std::set<std::string> getConsumers() const;

        // bool isSet();

        std::string getLogID() const override;
        std::string toString() const;
        std::string toLongString() const;

        static void reset();
    };

}

#endif // BARRIER_DEFINITION_HPP
