#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void removeDropOut(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void removeExtraScale(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void removeExtraBias(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void updateQuantParamsfromOpenVino(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void removeReshape(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(RemoveDropOut)
        .setFunc(removeDropOut)
        .setDescription(
            "Removes dropout layers from the network"
        );
        MV_REGISTER_PASS(removeExtraScale)
        .setFunc(removeExtraScale)
        .setDescription(
            "Removes dropout layers from the network"
        );
        MV_REGISTER_PASS(removeExtraBias)
        .setFunc(removeExtraBias)
        .setDescription(
            "Removes dropout layers from the network"
        );
        MV_REGISTER_PASS(updateQuantParamsfromOpenVino)
        .setFunc(updateQuantParamsfromOpenVino)
        .setDescription(
            "Update DPU task and output tensor with open vino quantization params"
        );
         MV_REGISTER_PASS(removeReshape)
        .setFunc(removeReshape)
        .setDescription(
            "Remove reshape"
        );

    }

}


mv::Data::OpListIterator linkNewOperationsRemove(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    while(opIt.parentsSize() > 1)
    {
        auto paramOp = opIt.leftmostParent();
        ++paramOp;
        om.removeOp(paramOp);
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}
void removeDropOut(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == "Dropout")
        {
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

            auto sourceTensor = parentOpIt->getOutputTensor(0);

            opIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, opIt);
            if (outputMemoryLocation.isForced())
            {
                opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            }
        }
    }
}

void removeExtraScale(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == "Scale")
        {
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

            auto sourceTensor = parentOpIt->getOutputTensor(0);

            opIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, opIt);
            if (outputMemoryLocation.isForced())
            {
                opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            }
        }
    }
}

void removeExtraBias(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == "Bias")
        {
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

            auto sourceTensor = parentOpIt->getOutputTensor(0);

            opIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, opIt);
            if (outputMemoryLocation.isForced())
            {
                opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            }
        }
    }
}

void removeReshape(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == "Reshape")
        {
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

            auto sourceTensor = parentOpIt->getOutputTensor(0);

            opIt = linkNewOperationsRemove(parentOpIt, sourceTensor, om, opIt);
            if (outputMemoryLocation.isForced())
            {
                opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            }
        }
    }
}

void updateQuantParamsfromOpenVino(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    using namespace mv;

    OpModel om(model);

    for (auto opIt = om.getInput(); opIt != om.opEnd(); ++opIt)
    {

        if (opIt->getOpType() == "Scale")
        {
            //Get Scale DPU parent
            auto dpuParent = opIt.leftmostParent();
            auto outputChannels = dpuParent->getOutputTensor()[0]->getShape()[IO_CHANNEL_DIMENSION];

            //Get scale const parent
            auto scaleConst = opIt.rightmostParent();

            //Get scale const data = shift
            auto type = scaleConst->getOutputTensor()[0]->getDType();
            auto shiftData = scaleConst->getOutputTensor()[0]->getDoubleData(); 

            //Get bias op after scale
            auto biasOp = opIt.leftmostChild();
            
            //Get bias Op const input
            auto biasConst = biasOp.rightmostParent();

            //Get bias const data = zeropoint
            auto zeroPointData = biasConst->getOutputTensor()[0]->getIntData();


            std::vector<double> min(outputChannels, 0);
            std::vector<double> max(outputChannels, 0);
            auto newQuant = mv::QuantizationParams(zeroPointData,shiftData ,min,max);
        
            //update quant params for op
            dpuParent->set<mv::QuantizationParams>("quantizationParams",newQuant);

            dpuParent->getOutputTensor()[0]->set<mv::QuantizationParams>("quantizationParams",newQuant);
        }
    }
}
