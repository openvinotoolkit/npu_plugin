#include "include/mcm/pass/pass_manager.hpp"



mv::PassManager::PassManager() :
ready_(false),
completed_(false),
model_(nullptr),
currentStage_(passFlow_.cend()),
currentPass_(adaptPassQueue_.end())
{

}

bool mv::PassManager::initialize(ComputationModel &model, const TargetDescriptor& targetDescriptor)
{
    
    targetDescriptor_ = targetDescriptor;
    adaptPassQueue_ = targetDescriptor_.adaptPasses();
    optPassQueue_ = targetDescriptor_.optPasses();
    finalPassQueue_ = targetDescriptor_.finalPasses();
    serialPassQueue_ = targetDescriptor_.serialPasses();
    validPassQueue_ = targetDescriptor_.validPasses();

    auto checkQueue = [this](const std::vector<std::string>& queue, PassGenre genre)
    {
        for (std::size_t i = 0; i < queue.size(); ++i)
        {

            pass::PassEntry *passPtr = pass::PassRegistry::instance().find(queue[i]);
            if (passPtr == nullptr)
            {
                reset();
                return false;
            }

            auto passGenres = passPtr->getGenre();
            if (passGenres.find(genre) == passGenres.end())
            {
                reset();
                return false;
            }

        }

        return true;

    };

    if (!checkQueue(adaptPassQueue_, PassGenre::Adaptation))
        return false;

    if (!checkQueue(optPassQueue_, PassGenre::Optimization))
        return false;

    if (!checkQueue(finalPassQueue_, PassGenre::Finalization))
        return false;

    if (!checkQueue(serialPassQueue_, PassGenre::Serialization))
        return false;

    if (!checkQueue(validPassQueue_, PassGenre::Validation))
        return false;

    if (!model.isValid())
    {
        reset();
        return false;
    }

    model_ = &model;
    ready_ = true;
    completed_ = false;
    currentStage_ = passFlow_.cbegin();
    currentPass_ = adaptPassQueue_.begin();
    return true;

}

void mv::PassManager::reset()
{

    model_ = nullptr;
    ready_ = false;
    completed_ = false;
    adaptPassQueue_.clear();
    optPassQueue_.clear();
    finalPassQueue_.clear();
    serialPassQueue_.clear();
    validPassQueue_.clear();
    currentStage_ = passFlow_.cbegin();
    currentPass_ = adaptPassQueue_.end();

}

bool mv::PassManager::ready() const
{
    return model_ && ready_;
}

bool mv::PassManager::completed() const
{
    return ready() && completed_;
}

std::pair<std::string, mv::PassGenre> mv::PassManager::step()
{

    while (!completed_ && currentStage_ != passFlow_.cend())
    {

        if (currentPass_ == currentStage_->second.cend())
        {
            
            ++currentStage_;
            if (currentStage_ == passFlow_.end())
            {
                completed_ = true;
                return std::pair<std::string, mv::PassGenre>("", PassGenre::Serialization);
            }

            currentPass_ = currentStage_->second.begin();
            
        }
        else
        {

            pass::PassRegistry::instance().find(*currentPass_)->run(*model_, targetDescriptor_);
            std::pair<std::string, mv::PassGenre> result({*currentPass_, currentStage_->first});
            currentPass_++;
            return result;

        }

    }

    return std::pair<std::string, mv::PassGenre>("", PassGenre::Adaptation);

}