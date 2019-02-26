#ifndef BARRIER_DEFINITION_HPP
#define BARRIER_DEFINITION_HPP

#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    class Barrier : public LogSender
    {
        int group_;
        int index_;        
        int numProducers_;
        int numConsumers_;

    public:
        Barrier(int group = -1, int index = -1, int numProducers = 0, int numConsumers = 0);

        int getGroup();
        int getIndex();

        int getNumProducers();
        int getNumConsumers();

        void setGroup();
        void setIndex();

        void addProducer();
        void addConsumer();
        void clear();

        // are these needed?
        // bool hasProducers();
        // bool hasConsumers();
        // bool isSet();

        std::string getLogID() const override;
        std::string toString() const;
    };

}

#endif // BARRIER_DEFINITION_HPP
