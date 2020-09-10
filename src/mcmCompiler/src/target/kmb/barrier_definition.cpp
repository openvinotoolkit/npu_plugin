#include "include/mcm/target/kmb/barrier_definition.hpp"

int mv::Barrier::barrierCounter_ = 0;

mv::Barrier::Barrier(int group, int index) :
group_(group),
index_(index),
barrierID_(barrierCounter_),
realBarrierIndex_(),
numProducers_(0),
numConsumers_(0),
producers_(),
consumers_()
{
    barrierCounter_++;
}

mv::Barrier::Barrier(int group, int index, std::set<std::string>& producers, std::set<std::string>& consumers) :
group_(group),
index_(index),
barrierID_(barrierCounter_),
realBarrierIndex_(),
numProducers_(producers.size()),
numConsumers_(consumers.size()),
producers_(producers),
consumers_(consumers)
{
    barrierCounter_++;
}

mv::Barrier::Barrier(std::set<std::string>& producers, std::set<std::string>& consumers) :
group_(-1),
index_(-1),
barrierID_(barrierCounter_),
realBarrierIndex_(),
numProducers_(producers.size()),
numConsumers_(consumers.size()),
producers_(producers),
consumers_(consumers)
{
    barrierCounter_++;
}

bool mv::Barrier::operator==(const Barrier& other)
{
    return barrierID_ == other.barrierID_;
}

bool mv::Barrier::operator!=(const Barrier& other)
{
    return barrierID_ != other.barrierID_;
}

int mv::Barrier::getNumProducers() const
{
    return numProducers_;
}

int mv::Barrier::getNumConsumers() const
{
    return numConsumers_;
}

void mv::Barrier::setNumProducers(int producers)
{
    numProducers_ = producers;
}

void mv::Barrier::setNumConsumers(int consumers)
{
    numConsumers_ = consumers;
}

void mv::Barrier::setGroup(int group)
{
    group_ = group;
}

void mv::Barrier::setIndex(int index)
{
    index_ = index;
}

int mv::Barrier::getID() const
{
    return barrierID_;
}

void mv::Barrier::setID(int id)
{
    barrierID_ = id;
}

int mv::Barrier::getIndex() const
{
    return index_;
}

int mv::Barrier::getGroup() const
{
    return group_;
}

void mv::Barrier::addProducer(const std::string& producer)
{
    if (!producers_.count(producer))
    {
        producers_.insert(producer);
        numProducers_++;
    }
}

void mv::Barrier::addConsumer(const std::string& consumer)
{
    if (!consumers_.count(consumer))
    {
        consumers_.insert(consumer);
        numConsumers_++;
    }
}

void mv::Barrier::removeProducer(const std::string& producer)
{
    auto it = producers_.find(producer);

    if (it == producers_.end())
        return;

    producers_.erase(it);
    numProducers_--;
}

void mv::Barrier::removeConsumer(const std::string& consumer)
{
    auto it = consumers_.find(consumer);

    if (it == consumers_.end())
        return;

    consumers_.erase(it);
    numConsumers_--;
}

void mv::Barrier::clear()
{
    numProducers_ = 0;
    numConsumers_ = 0;
    group_ = -1;
    index_ = -1;
    producers_.clear();
    consumers_.clear();
}

bool mv::Barrier::hasProducers() const
{
    return !producers_.empty();
}

bool mv::Barrier::hasConsumers() const
{
    return !consumers_.empty();
}

std::set<std::string> mv::Barrier::getProducers() const
{
    return producers_;
}

std::set<std::string> mv::Barrier::getConsumers() const
{
    return consumers_;
}

std::string mv::Barrier::toString() const
{
    std::string output = "";

    output += "group: " + std::to_string(group_) + " | ";
    output += "index: " + std::to_string(index_) + " | ";
    output += "nProd: " + std::to_string(numProducers_) + " | ";
    output += "nCons: " + std::to_string(numConsumers_) + "\n";

    return output;
}

std::string mv::Barrier::toLongString() const
{
    std::string output = "";

    output += "id   : " + std::to_string(barrierID_) + " | ";
    output += "group: " + std::to_string(group_) + " | ";
    output += "index: " + std::to_string(index_) + " | ";
    output += "nProd: " + std::to_string(numProducers_) + " | ";
    output += "nCons: " + std::to_string(numConsumers_) + "\n";
    output += "producers = ";
    for (auto prod: producers_)
        output += prod + ", ";

    output += "\n";

    output += "consumers = ";
    for (auto cons: consumers_)
        output += cons + ", ";

    output += "\n";

    return output;
}

std::string mv::Barrier::getLogID() const
{
    return "Barrier:" + toString();
}

void mv::Barrier::reset()
{
    barrierCounter_ = 0;
}
