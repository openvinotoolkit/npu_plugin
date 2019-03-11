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
        int numProducers_;
        int numConsumers_;
        std::unordered_set<std::string> producers_; // names of the input ops
        std::unordered_set<std::string> consumers_; // names of the op that consume this barrier

    public:
        Barrier(int group = -1, int index = -1);
        Barrier(int group, int index, std::unordered_set<std::string>& producers, std::unordered_set<std::string>& consumers);
        Barrier(std::unordered_set<std::string>& producers, std::unordered_set<std::string>& consumers);

        int getGroup() const;
        int getIndex() const;

        int getNumProducers() const;
        int getNumConsumers() const;

        void setGroup(int group);
        void setIndex(int index);

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
    };

}

#endif // BARRIER_DEFINITION_HPP
