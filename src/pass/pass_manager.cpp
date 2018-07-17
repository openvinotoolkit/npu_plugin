#include "include/mcm/pass/pass_manager.hpp"

mv::PassManager::PassManager() :
model_(nullptr),
ready_(false),
currentStage_(PassGenre::Adaptation),
previousStage_(PassGenre::Adaptation),
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

            base::PassEntry *passPtr = base::PassRegistry::instance().find(queue[i]);
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
    currentStage_ = PassGenre::Adaptation;
    previousStage_ = PassGenre::Adaptation;
    currentPass_ = adaptPassQueue_.begin();
    return true;

}

void mv::PassManager::reset()
{

    model_ = nullptr;
    ready_ = false;
    adaptPassQueue_.clear();
    optPassQueue_.clear();
    finalPassQueue_.clear();
    serialPassQueue_.clear();
    validPassQueue_.clear();
    currentStage_ = PassGenre::Adaptation;
    previousStage_ = PassGenre::Adaptation;
    currentPass_ = adaptPassQueue_.end();

}

bool mv::PassManager::ready()
{
    return model_ && ready_;
}

std::pair<std::string, mv::PassGenre> mv::PassManager::step()
{

    auto execAndNext = [this](const std::vector<string>& queue, PassGenre nextStage, std::vector<string>& nextQueue)
    {

        if (currentPass_ == queue.end())
        {
            currentStage_ = nextStage;
            currentPass_ = nextQueue.begin();
            return std::pair<std::string, mv::PassGenre>("", currentStage_);
        }
        else
        {
            base::PassRegistry::instance().find(*currentPass_)->run(*model_, targetDescriptor_);
            std::pair<std::string, mv::PassGenre> result({*currentPass_, currentStage_});
            currentPass_++;
            return result;
        }

    };

    switch (currentStage_)
    {

        case PassGenre::Adaptation:
            if (currentPass_ == adaptPassQueue_.end())
            {
                currentStage_ = PassGenre::Optimization;
                currentPass_ = optPassQueue_.begin();
            }
            else
            {
                base::PassRegistry::instance().find(*currentPass_)->run(*model_, targetDescriptor_);
                std::pair<std::string, mv::PassGenre> result({*currentPass_, PassGenre::Adaptation});
                currentPass_++;
                return result;
            }
        
        case PassGenre::Optimization:
            if (currentPass_ == optPassQueue_.end())
            {
                currentStage_ = PassGenre::Finalization;
                currentPass_ = finalPassQueue_.begin();
            }
            else
            {
                base::PassRegistry::instance().find(*currentPass_)->run(*model_, targetDescriptor_);
                std::pair<std::string, mv::PassGenre> result({*currentPass_, PassGenre::Optimization});
                currentPass_++;
                return result;
            }
        
        case PassGenre::Finalization:
            if (currentPass_ == finalPassQueue_.end())
            {
                currentStage_ = PassGenre::Serialization;
                currentPass_ = serialPassQueue_.begin();
            }
            else
            {
                base::PassRegistry::instance().find(*currentPass_)->run(*model_, targetDescriptor_);
                std::pair<std::string, mv::PassGenre> result({*currentPass_, PassGenre::Finalization});
                currentPass_++;
                return result;
            }

        case PassGenre::Serialization:
            if (currentPass_ == serialPassQueue_.end())
            {
                return {"", PassGenre::Serialization};
            }
            else
            {
                base::PassRegistry::instance().find(*currentPass_)->run(*model_, targetDescriptor_);
                std::pair<std::string, mv::PassGenre> result({*currentPass_, PassGenre::Serialization});
                currentPass_++;
                return result;
            }

        case PassGenre::Validation:
            if (currentPass_ == validPassQueue_.end())
            {
                switch (previousStage_)
                {
                    case PassGenre::Adaptation:
                        currentStage_ = PassGenre::Optimization;
                        currentPass_ = optPassQueue_.begin();


                }
            }
        

    }

}