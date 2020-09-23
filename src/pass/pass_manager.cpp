#include "include/mcm/pass/pass_manager.hpp"

mv::PassManager::PassManager() :
initialized_(false),
completed_(false),
running_(false),
model_(nullptr),
compOutput_("CompilationOutput")
{

}

bool mv::PassManager::initialize(ComputationModel &model, const TargetDescriptor& targetDescriptor, const mv::CompilationDescriptor& compDescriptor)
{

    if (running_)
        return false;

    reset();

    targetDescriptor_ = targetDescriptor;

    if (!model.isValid())
    {
        log(mv::Logger::MessageType::Error, "Invalid input model - might miss input or output or be disjoint");
        reset();
        return false;
    }

    model_ = &model;
    initialized_ = true;
    completed_ = false;
    running_ = false;
    compDescriptor_ = compDescriptor;
    compOutput_.clear();
    return true;

}

void mv::PassManager::reset()
{

    model_ = nullptr;
    initialized_ = false;
    completed_ = false;
    running_ = false;
    passList_.clear();
    currentPass_ = passList_.begin();
    targetDescriptor_ = TargetDescriptor();
    compDescriptor_ = CompilationDescriptor();
    compOutput_.clear();

}

bool mv::PassManager::initialized() const
{
    return model_ && initialized_;
}

bool mv::PassManager::completed() const
{
    return initialized() && !running_ && completed_;
}

void mv::PassManager::loadPassList(const std::vector<mv::Element>& passList)
{
    passList_ = passList;
    currentPass_ = passList_.begin();
}

mv::Element& mv::PassManager::step()
{

    if (!running_)
    {
        if (!validDescriptors())
            throw RuntimeError(*this, "Invalid descriptor");

        running_ = true;

        if (currentPass_ == passList_.end())
        {
            completed_ = true;
            running_ = false;
            compOutput_.set<bool>("finished", true);
            return compOutput_;
        }

        // Initialize pass execution...
        currentPass_ = passList_.begin();
        compOutput_.clear();
        compOutput_.set<bool>("finished", false);
        compOutput_.set<std::vector<Element>>("passes", {});
    }

    if (completed_ || currentPass_ == passList_.end())
    {
        completed_ = true;
        running_ = false;
        compOutput_.set<bool>("finished", true);
        return compOutput_;
    }

    auto passPtr = pass::PassRegistry::instance().find(currentPass_->getName());
    if (passPtr == nullptr) {
        throw RuntimeError(*this, "Pass not found in the registry");
    }

    compOutput_.get<std::vector<Element>>("passes").push_back(Element(passPtr->getName()));
    Element& lastPassOutput = compOutput_.get<std::vector<Element>>("passes").back();

    auto passElem = *currentPass_;

    if (model_->getGlobalConfigParams()->hasAttr("disabled_labels"))
    {
        auto disabledLabels = model_->getGlobalConfigParams()->get<std::vector<std::string>>("disabled_labels");
        for (auto it = disabledLabels.begin(); it != disabledLabels.end(); ++it)
        {
            if (passPtr->hasLabel(*it))
            {
                log(Logger::MessageType::Info, "Skipping pass " + passPtr->getName() + " with label \"" + *it + "\"");
                currentPass_++;
                return compOutput_;
            }
        }
    }
        
    log(Logger::MessageType::Info, "Starting pass " + passPtr->getName());
    passPtr->run(*model_, targetDescriptor_, passElem, lastPassOutput);
    log(Logger::MessageType::Info, "Finished pass " + passPtr->getName());
    currentPass_++;

    return compOutput_;

}

bool mv::PassManager::validPassArgs() const
{
    if (passList_.empty())
    {
        log(Logger::MessageType::Error, "Pass list empty, cannot validate pass arguments.");
        return false;
    }

    for (auto p: passList_)
    {
        auto passEntry = pass::PassRegistry::instance().find(p.getName());
        if (!passEntry)
            throw RuntimeError(*this, "Queried pass " + p.getName() + " not found in the pass registry");
        for (auto reqdArg: passEntry->getArgs())
        {
            if (!p.hasAttr(reqdArg.first))
            {
                log(Logger::MessageType::Error, "Missing required pass arg " + reqdArg.first + " for PassEntry " + p.getName());
                return false;
            }
        }
    }

    return true;
}

bool mv::PassManager::validDescriptors() const
{

    if (!initialized())
    {
        log(Logger::MessageType::Error, "Pass manager not initialized");
        return false;
    }

    if (targetDescriptor_.getTarget() == Target::Unknown)
    {
        log(Logger::MessageType::Error, "Target descriptor has an undefined target");
        return false;
    }

    if (!validPassArgs()) {
        log(Logger::MessageType::Error, "Required pass arguments not present");
        return false;
    }

    return true;

}

std::string mv::PassManager::getLogID() const
{
    return "PassManager";
}
