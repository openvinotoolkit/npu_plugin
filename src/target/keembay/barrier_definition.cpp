#include "include/mcm/target/keembay/barrier_definition.hpp"

mv::Barrier::Barrier(int group, int index, int numProducers, int numConsumers) :
group_(group),
index_(index),
numProducers_(numProducers),
numConsumers_(numConsumers)
{

}

int mv::Barrier::getNumProducers()
{
    return numProducers_;
}

int mv::Barrier::getNumConsumers()
{
    return numConsumers_;
}

int mv::Barrier::getIndex()
{
    return index_;
}

int mv::Barrier::getGroup()
{
    return group_;
}

// Yet to be implemented
void mv::Barrier::addProducer()
{
    numProducers_++;
}

void mv::Barrier::addConsumer() {
    numConsumers_++;
}

void mv::Barrier::clear()
{
    numProducers_ = 0;
    numConsumers_ = 0;
    group_ = -1;
    index_ = -1;
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

std::string mv::Barrier::getLogID() const
{
    return "Barrier:" + toString();
}