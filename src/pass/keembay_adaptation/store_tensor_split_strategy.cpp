#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "meta/include/mcm/op_model.hpp"
#include <regex>

static void storeLayerSplitStrategyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
static void storeTensorPlacementFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&,mv::json::Object&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(StoreLayerSplitStrategy)
        .setFunc(storeLayerSplitStrategyFcn)
        .setDescription(
            "This pass applies tensor splitting strategies."
        );

        MV_REGISTER_PASS(StoreTensorPlacement)
        .setFunc(storeTensorPlacementFcn)
        .setDescription(
            "This pass applies the memory location overrides for the Tensors from the JSON file."
        );
    }
}

void storeStrategy(mv::Data::OpListIterator& it, int numClusters, std::vector<mv::Element>& strategyList)
{
    for (auto s: strategyList)
    {
        std::string& name_filter = s.get<std::string>("name_filter");
        std::regex exp(name_filter);
        if (std::regex_match(it->getName(), exp))
        {
            it->set<std::string>("splitStrategy", s.get<std::string>("strategy"));
        }
    }
}

void storeLayerSplitStrategyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    auto globalParams = model.getGlobalConfigParams();

    if (!globalParams->hasAttr("split_strategy"))
    {
        pass.log(mv::Logger::MessageType::Info, "No custom splitting strategy provided, exiting...");
        return;
    }

    auto strategyList = globalParams->get<std::vector<mv::Element>>("split_strategy");
    auto numClusters = globalParams->get("Number_of_Clusters");

    mv::OpModel om(model);

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto opType = opIt->getOpType();
        if (!(opType == "Input" || opType == "Output"))
            storeStrategy(opIt, numClusters, strategyList);
    }

    pass.log(mv::Logger::MessageType::Info, "----splitting strategies for individual layers----");
    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->hasAttr("splitStrategy"))
        {
            pass.log(mv::Logger::MessageType::Info, "op: " + opIt->getName() +
                        " | strategy = " + opIt->get<std::string>("splitStrategy"));
        }
    }
}

void storeTensorPlacementFcn(const mv::pass::PassEntry& pass,
                                mv::ComputationModel& model,
                                mv::TargetDescriptor&,
                                mv::Element&,mv::json::Object&)
{
    //mv::Logger::setVerboseLevel(mv::VerboseLevel::Debug);

    auto globalParams = model.getGlobalConfigParams();
    mv::OpModel om(model);

    if (globalParams->hasAttr("tensor_placement_override"))
    {
        auto placementOverrideList = globalParams->get<std::vector<mv::Element>>("tensor_placement_override");


        for (auto tensorIt = om.tensorBegin() ; tensorIt != om.tensorEnd() ; ++tensorIt)
        {
            auto parentOp = om.getSourceOp(tensorIt);

            if(parentOp != om.opEnd())
                if(parentOp->getOpType() == "Input")
                {
                    continue;
                }

            bool found = false;
            for( auto s : placementOverrideList )
            {
                std::string& nameFilter = s.get<std::string>("name_filter");
                std::string& memLocation = s.get<std::string>("mem_location");
                bool forced = s.hasAttr("force");

                std::regex exp(nameFilter);
                if (std::regex_match(tensorIt->getName(),exp))
                {
                    found = true;
                    mv::Tensor::MemoryLocation location(memLocation,forced);
                    tensorIt->set<mv::Tensor::MemoryLocation>("Location",location);
                    pass.log(mv::Logger::MessageType::Info,"setting tensor " +
                                tensorIt->getName() + " as " + location.toString());
                }
            }

            if((not found) and (not tensorIt->hasAttr("Location")))
            {
                tensorIt->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::DEFAULT);
                pass.log(mv::Logger::MessageType::Info,"tensor " + tensorIt->getName() + "not found. setting to DEFAULT");
            }
        }
    }
    else
    {
        pass.log(mv::Logger::MessageType::Info, "No tensor placement override provided, exiting...");

    }

    auto input = om.getInput();
    auto inputTensors = input->getOutputTensor();
    for (auto tensor : inputTensors)
    {
        if(!tensor->get<mv::Tensor::MemoryLocation>("Location").isDefault())
        {
            pass.log(mv::Logger::MessageType::Warning, "Found InputTensor " +
                        tensor->getName() + " description location in JSON. Will override with INPUT");
        }
        pass.log(mv::Logger::MessageType::Warning,"Found OutputTensor " +
                        tensor->getName() + " current location is " + tensor->get<mv::Tensor::MemoryLocation>("Location").toString() + " override with OUTPUT");
        //mark location forced so any adaptation pass will inheret it
        tensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation("INPUT", true));
    }

    auto output = om.getOutput();
    auto outputTensors = output->getInputTensor();

    for ( auto tensor : outputTensors)
    {
        if(!tensor->get<mv::Tensor::MemoryLocation>("Location").isDefault())
        {
            pass.log(mv::Logger::MessageType::Warning,"Found OutputTensor " +
                        tensor->getName() + " description location in JSON. Will override with OUTPUT");
        }
        pass.log(mv::Logger::MessageType::Warning,"Found OutputTensor " +
                        tensor->getName() + " current location is " + tensor->get<mv::Tensor::MemoryLocation>("Location").toString() + " override with OUTPUT");
        //mark location forced so any adaptation pass will inheret it
        tensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation("OUTPUT", true));
    }

    //if JSON wants us to override default location for tensors:
    if(globalParams->hasAttr("default_tensor_placement"))
    {
        std::string& defaultPlace = globalParams->get<std::string>("default_tensor_placement");
        for (auto tensorIt = om.tensorBegin() ; tensorIt != om.tensorEnd() ; ++tensorIt)
        {
            if(tensorIt->get<mv::Tensor::MemoryLocation>("Location").isDefault())
            {
                tensorIt->get<mv::Tensor::MemoryLocation>("Location").relocate(defaultPlace);
                pass.log(mv::Logger::MessageType::Warning,"Found OutputTensor " +
                        tensorIt->getName() + " overriding all to default placement " + defaultPlace);
            }
        }
    }
    //mv::Logger::setVerboseLevel(mv::VerboseLevel::Warning);
}
