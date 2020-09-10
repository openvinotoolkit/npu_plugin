#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void alignSliceOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AlignSliceOps)
            .setFunc(alignSliceOpsFcn)
            .setDescription(
                "Aligns slice ops in the correct shape and order required by Kmb");
    }
}

void alignSliceOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto sliceOps = om.getOps("Slice");

    for(auto op: sliceOps)
    {

        if (!op->getInputTensor(0)->isPopulated())
        {
            pass.log(mv::Logger::MessageType::Debug, "input tensor not populated for slice of op " + op->getName() + "...not aligning");
            continue;
        }

        auto opId = op->get<unsigned>("opId");

        pass.log(mv::Logger::MessageType::Debug, "Aligning sliceOp " + op->getName());

        auto inputMemoryLocation = op->getInputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        auto outputMemoryLocation = op->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

        auto origBegin = op->get<mv::Shape>("begin");
        auto origSize  = op->get<mv::Shape>("size");

        auto input = op->getInputTensor(0);
        auto output = op->getOutputTensor(0);
        auto outputShape = output->getShape();
        auto outputName = op->getName();
        auto outputDType = output->getDType();
        mv::QuantizationParams quantParams = {{},{},{},{}};
        if(input->hasAttr("quantParams"))
            quantParams = input->get<mv::QuantizationParams>("quantParams");

        auto inputChannels = outputShape[mv::KERNEL_INPUT_CHANNELS];
        auto outputWidth = outputShape[mv::KERNEL_WIDTH];
        auto outputHeight = outputShape[mv::KERNEL_HEIGHT];

        auto outputChannels = outputShape[mv::KERNEL_OUTPUT_CHANNELS];
        auto weightSetDimension = outputWidth * outputHeight * inputChannels;

        // Not sure if this will work for slice...
        auto weightSetDimensionPadded = mv::round_up(weightSetDimension, 16);
        auto paddingDifference = weightSetDimensionPadded - weightSetDimension;

        mv::Shape newShape({weightSetDimensionPadded, 1, 1, outputChannels});

        auto newSlice = om.slice(input,
                                origBegin,
                                newShape,
                                quantParams,
                                op->getName() + "_Aligned");

        std::string newOutputName = outputName;
        if(paddingDifference != 0)
            newOutputName = mv::createAlignWeightSetConstantName(outputName);

        auto ctlFlow = cm.switchContext(op);
        std::vector<mv::Control::OpListIterator> inputControlFlows;
        for (auto parentOp = ctlFlow.leftmostParent(); parentOp != cm.opEnd(); ++parentOp)
        {
            inputControlFlows.push_back(parentOp);
            pass.log(mv::Logger::MessageType::Debug, "FOUND slice parent " + parentOp->getName() + " of " + op->getName());
        }

        auto outputDataFlows = mv::getOutputDataFlow(om, op);

        auto newOp = om.getSourceOp(newSlice);
        newOp->set<unsigned>("opId", opId);

        mv::setOutputDataFlow(om, newSlice, outputDataFlows);
        for (auto cp : inputControlFlows)
        {
            cm.defineFlow(om.switchContext(cp),om.getSourceOp(newSlice));
            pass.log(mv::Logger::MessageType::Debug, "ADDED CONTROL FLOW from " + cp->getName() + " to " + om.getSourceOp(newSlice)->getName());
        }

        newSlice->set<mv::Shape>("OriginalShape", outputShape);
        newSlice->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);

        for(auto& flowPair: outputDataFlows)
        {
            flowPair.first->set<std::array<unsigned short, 2>>("kSize", {outputWidth, outputHeight});
            flowPair.first->set<unsigned>("inputChannels", inputChannels);
            flowPair.first->set<unsigned>("outputChannels", outputChannels);
            flowPair.first->set<mv::QuantizationParams>("quantParams", quantParams);
        }
    }
}
