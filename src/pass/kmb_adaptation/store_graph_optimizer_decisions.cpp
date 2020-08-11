#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include <regex>

static void storeLayerSplitStrategyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void storeTensorPlacementFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void storeDilationConcatsDDRFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
static void solveDilatedSlicingFcn(const mv::pass::PassEntry&, mv::ComputationModel& model);
static void validateDilationSubConvolutions(const mv::pass::PassEntry&, mv::ComputationModel& model);
static void storeLayerSparsityStrategyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model);
static void storeGraphOptimizerDecisions(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(StoreGraphOptimizerDecisions)
        .setFunc(storeGraphOptimizerDecisions)
        .setDescription(
            "This pass stores to the graph the output of graph optimizer."
        );
    }
}

void storeStrategy(mv::Data::OpListIterator& opIt, std::vector<mv::Element>& strategyList)
{
    for (auto s: strategyList)
    {
        std::string& name_filter = s.get<std::string>("name_filter");
        std::regex exp(name_filter);
        if (std::regex_match(opIt->getName(), exp))
        {
            auto strategy = s.get<std::string>("strategy");
            opIt->set<std::string>("splitStrategy", strategy);
            if(strategy == "SplitOverK" || strategy == "HKSwitch")
                opIt->set<bool>("multiCast", true);
            else
                opIt->set<bool>("multiCast", false);
            break; //the operation can have only one strategy, and the nn filter (operation) was found
        }
    }
}


void storeGraphOptimizerDecisions(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    storeLayerSplitStrategyFcn(pass, model);
    storeLayerSparsityStrategyFcn(pass, model);
    storeTensorPlacementFcn(pass, model);
    //NOTE: The idea of that pass is that for the dilation convolution all the dmas of the subconvolutions
    //need to write on one master buffer located or in the ddr or in the output location
    storeDilationConcatsDDRFcn(pass, model);
    solveDilatedSlicingFcn(pass, model);
    validateDilationSubConvolutions(pass, model);
}

void storeLayerSplitStrategyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    auto globalParams = model.getGlobalConfigParams();

    if (!globalParams->hasAttr("split_strategy"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No custom splitting strategy provided, exiting...");
        return;
    }

    auto strategyList = globalParams->get<std::vector<mv::Element>>("split_strategy");

    mv::OpModel om(model);

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto opType = opIt->getOpType();
        if (!(opType == "Output"))
            storeStrategy(opIt, strategyList);
    }

    pass.log(mv::Logger::MessageType::Debug, "----splitting strategies for individual layers----");
    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        if (opIt->hasAttr("splitStrategy"))
        {
            pass.log(mv::Logger::MessageType::Debug, "op: " + opIt->getName() +
                        " | strategy = " + opIt->get<std::string>("splitStrategy"));
        }
    }
}

void storeLayerSparsityStrategyFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    auto globalParams = model.getGlobalConfigParams();

    if (!globalParams->hasAttr("sparsity_strategy"))
    {
        pass.log(mv::Logger::MessageType::Debug, "No custom sparsity strategy provided, exiting...");
        return;
    }

    auto strategyList = globalParams->get<std::vector<mv::Element>>("sparsity_strategy");

    mv::OpModel om(model);

    for (auto opIt = om.opBegin(); opIt != om.opEnd(); ++opIt)
    {
        auto opType = opIt->getOpType();
        bool strategyFound = false;
        if (opType != "Output" && opType != "Input")
        {
            for (auto s: strategyList)
            {
                std::string& name_filter = s.get<std::string>("name_filter");
                std::regex exp(name_filter);
                if (std::regex_match(opIt->getName(), exp))
                {
                    bool inputActivationSparsity, outputActivationSparsity, weightsSparsity = false;
                    if(s.hasAttr("inputActivationSparsity"))
                        inputActivationSparsity = s.get<bool>("inputActivationSparsity");
                    if(s.hasAttr("outputActivationSparsity"))
                        outputActivationSparsity = s.get<bool>("outputActivationSparsity");
                    if(s.hasAttr("weightsSparsity"))
                        weightsSparsity = s.get<bool>("weightsSparsity");

                    opIt->set<bool>("inputActivationSparsity", inputActivationSparsity);
                    opIt->set<bool>("outputActivationSparsity", outputActivationSparsity);
                    opIt->set<bool>("weightsSparsity", weightsSparsity);
                    strategyFound = true;
                }
            }
            if(!strategyFound)
            {
                opIt->set<bool>("inputActivationSparsity", false);
                opIt->set<bool>("outputActivationSparsity", false);
                opIt->set<bool>("weightsSparsity", false);
            }
        }
    }
}

void storeTensorPlacementFcn(const mv::pass::PassEntry& pass,
                                mv::ComputationModel& model)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
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
                    pass.log(mv::Logger::MessageType::Debug,"setting tensor " +
                                tensorIt->getName() + " as " + location.toString());
                }
            }

            if((not found) and (not tensorIt->hasAttr("Location")))
            {
                tensorIt->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::DEFAULT);
                pass.log(mv::Logger::MessageType::Debug,"tensor " + tensorIt->getName() + "not found. setting to DEFAULT");
            }
        }
    }
    else
    {
        pass.log(mv::Logger::MessageType::Debug, "No tensor placement override provided, exiting...");

    }

    auto input = om.getInput();
    auto inputTensors = input->getOutputTensor();
    for (auto tensor : inputTensors)
    {
        if(!tensor->get<mv::Tensor::MemoryLocation>("Location").isDefault())
        {
            pass.log(mv::Logger::MessageType::Debug, "Found InputTensor " +
                        tensor->getName() + " description location in JSON. Will override with INPUT");
        }
        pass.log(mv::Logger::MessageType::Debug,"Found OutputTensor " +
                        tensor->getName() + " current location is " + tensor->get<mv::Tensor::MemoryLocation>("Location").toString() + " override with OUTPUT");
        //mark location forced so any adaptation pass will inheret opIt
        tensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation("INPUT", true));
    }

    auto implicitRInputSlice = om.getOps("ImplicitInputSlice");
    for (std::size_t i = 0; i < implicitRInputSlice.size(); i++)
    {
        auto inTensors = implicitRInputSlice[i]->getOutputTensor();
        for (auto tensor : inTensors)
        {
            if(!tensor->get<mv::Tensor::MemoryLocation>("Location").isDefault())
            {
                pass.log(mv::Logger::MessageType::Warning, "Found InputTensor " +
                            tensor->getName() + " description location in JSON. Will override with INPUT");
            }
            pass.log(mv::Logger::MessageType::Warning,"Found OutputTensor " +
                            tensor->getName() + " current location is " + tensor->get<mv::Tensor::MemoryLocation>("Location").toString() + " override with OUTPUT");
            //mark location forced so any adaptation pass will inheret opIt
            tensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation("INPUT", true));
        }
    }
    auto output = om.getOutput();
    auto outputTensors = output->getInputTensor();

    for ( auto tensor : outputTensors)
    {
        if(!tensor->get<mv::Tensor::MemoryLocation>("Location").isDefault())
        {
            pass.log(mv::Logger::MessageType::Debug,"Found OutputTensor " +
                        tensor->getName() + " description location in JSON. Will override with OUTPUT");
        }
        pass.log(mv::Logger::MessageType::Debug,"Found OutputTensor " +
                        tensor->getName() + " current location is " + tensor->get<mv::Tensor::MemoryLocation>("Location").toString() + " override with OUTPUT");
        //mark location forced so any adaptation pass will inheret opIt
        tensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation("OUTPUT", true));
    }

    auto implicitUnionOps = om.getOps("ImplicitUnion");
    for ( auto implicitUnion : implicitUnionOps)
    {
        auto inTensors = implicitUnion->getInputTensor();
        for (auto tensor : inTensors)
        {
            if(!tensor->get<mv::Tensor::MemoryLocation>("Location").isDefault())
            {
                pass.log(mv::Logger::MessageType::Warning, "Found InputTensor " +
                            tensor->getName() + " description location in JSON. Will override with OUTPUT");
            }
            pass.log(mv::Logger::MessageType::Warning,"Found InputTensor " +
                            tensor->getName() + " current location is " + tensor->get<mv::Tensor::MemoryLocation>("Location").toString() + " override with OUTPUT");
            //mark location forced so any adaptation pass will inheret opIt
            tensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation("OUTPUT", true));
        }
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
                pass.log(mv::Logger::MessageType::Debug,"Found OutputTensor " +
                        tensorIt->getName() + " overriding all to default placement " + defaultPlace);
            }
        }
    }
    //mv::Logger::setVerboseLevel(mv::VerboseLevel::Warning);
}

mv::Data::OpListIterator findNextNotImplicitOp(mv::DataModel &dataModel, const mv::Data::OpListIterator &op)
{
    auto sinkOp = mv::findSinkLayers(dataModel, op->getOutputTensor()[0])[0];
    while(sinkOp->isImplicit())
    {
        sinkOp = mv::findSinkLayers(dataModel, sinkOp->getOutputTensor()[0])[0];
    }
    return sinkOp;

}

void setConcatTensorsLocation(const mv::Data::OpListIterator &op, mv::Tensor::MemoryLocation memoryLocation)
{
    for (auto inputTensor : op->getInputTensor())
        inputTensor->set<mv::Tensor::MemoryLocation>("Location", memoryLocation);
    op->getOutputTensor()[0]->set<mv::Tensor::MemoryLocation>("Location", memoryLocation);
}

void storeDilationConcatsDDRFcn(const mv::pass::PassEntry&,
                                mv::ComputationModel& model)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto sortedOps = om.topologicalSort();
    bool outputAfterDilatedConcat = false;
    for ( auto concat : sortedOps)
    {
        if (concat->getOpType() == "Concat" ||
                concat->getOpType() == "ImplicitConcat")
        {
            if (concat->hasAttr("dilatedWidthConcat") &&
                    concat->get<bool>("dilatedWidthConcat"))
            {
                auto nextNotImplicitOp = findNextNotImplicitOp(dm, concat);
                auto previousSubConv = om.getSourceOp(concat->getInputTensor()[0]);
                if (nextNotImplicitOp->getOpType() == "Output")
                {
                    if ((previousSubConv->get<mv::DType>("dType") == mv::DType("UInt8") &&
                            (nextNotImplicitOp->get<mv::DType>("precision") == mv::DType("Default") ||
                             nextNotImplicitOp->get<mv::DType>("precision") == mv::DType("UInt8")))
                             ||
                             (previousSubConv->get<mv::DType>("dType") == mv::DType("Float16") &&
                              nextNotImplicitOp->get<mv::DType>("precision") == mv::DType("Float16")))
                    {
                        outputAfterDilatedConcat = true;
                        setConcatTensorsLocation(concat, mv::Tensor::MemoryLocation::OUTPUT);
                    }
                    else
                        setConcatTensorsLocation(concat, mv::Tensor::MemoryLocation::DDR);
                }
                else
                {
                    setConcatTensorsLocation(concat, mv::Tensor::MemoryLocation::DDR);
                }
            }
            else if (concat->hasAttr("joinSimulation") && concat->get<bool>("joinSimulation"))
            {
                if (outputAfterDilatedConcat)
                    for (auto inputTensor : concat->getInputTensor())
                        inputTensor->set<mv::Tensor::MemoryLocation>("Location", mv::Tensor::MemoryLocation::OUTPUT);
                else
                    setConcatTensorsLocation(concat, mv::Tensor::MemoryLocation::DDR);
            }
        }
    }
}

void solveDilatedSlicingFcn(const mv::pass::PassEntry&,
                                mv::ComputationModel& model)
{
    mv::OpModel om(model);
    auto convs = om.getOps("Conv");
    auto globalParams = model.getGlobalConfigParams();
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    for (auto layerNameStrategy : strategyList)
    {
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        for (auto conv : convs)
        {
            bool isDilatedConv = conv->hasAttr("DilatedSubConv") && conv->get<bool>("DilatedSubConv");
            if (isDilatedConv && nodeName == conv->getName())
            {
                auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");
                if (streaming_strategy[1].get<int>("H") > 1)
                    conv->set<bool>("slicedInput3DDMA", true);
            }
        }
    }
}

void validateDilationSubConvolutions(const mv::pass::PassEntry&,
                                mv::ComputationModel& model)
{
    mv::OpModel om(model);
    auto convs = om.getOps("Conv");
    auto globalParams = model.getGlobalConfigParams();
    std::map<std::string, std::set<std::vector<mv::Element>>> subdilationsStrategies;
    auto strategyList = globalParams->get<std::vector<mv::Element>>("streaming_strategy");
    for (auto layerNameStrategy : strategyList)
    {
        std::string nodeName = layerNameStrategy.get<std::string>("name_filter");
        std::set<std::vector<mv::Element>> setOfStrategies;
        for (auto conv : convs)
        {
            bool isDilatedConv = conv->hasAttr("DilatedSubConv") && conv->get<bool>("DilatedSubConv");
            if (isDilatedConv && nodeName == conv->getName())
            {
                auto streaming_strategy = layerNameStrategy.get<std::vector<mv::Element>>("splits");
                setOfStrategies.insert(streaming_strategy);
                subdilationsStrategies.insert(std::make_pair(conv->get<std::string>("parentOp"),
                                                             setOfStrategies));
            }
        }
        auto it = subdilationsStrategies.begin();
        while (it != subdilationsStrategies.end())
        {
            auto setOfDilationStrategies = it->second;
            //NOTE: Subdilation convolutions should have same strategy,
            //if not disable the optimization of streaming with same weigths
            assert (setOfDilationStrategies.size() == 1);
            it++;
        }
    }
}
