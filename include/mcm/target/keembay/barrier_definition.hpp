#ifndef BARRIER_DEFINITION_HPP
#define BARRIER_DEFINITION_HPP

#include <string>
#include <unordered_set>

#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    class Barrier : public LogSender
    {
        int group_;
        int index_;
        int barrierID_;
        int numProducers_;
        int numConsumers_;
        std::unordered_set<std::string> producers_; // names of the input ops
        std::unordered_set<std::string> consumers_; // names of the op that consume this barrier
        static int barrierCounter_;

    public:
        Barrier(int group = -1, int index = -1);
        Barrier(int group, int index, std::unordered_set<std::string>& producers, std::unordered_set<std::string>& consumers);
        Barrier(std::unordered_set<std::string>& producers, std::unordered_set<std::string>& consumers);
        bool operator==(const Barrier& other);
        bool operator!=(const Barrier& other);

        int getGroup() const;
        int getIndex() const;
        int getID() const;

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

        bool hasProducers();
        bool hasConsumers();

        std::unordered_set<std::string> getProducers() const;
        std::unordered_set<std::string> getConsumers() const;

        // bool isSet();

        std::string getLogID() const override;
        std::string toString() const;

        static void reset();
    };

}

#endif // BARRIER_DEFINITION_HPP
