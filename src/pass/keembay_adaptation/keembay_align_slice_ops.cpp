#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void alignSliceOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AlignSliceOps)
            .setFunc(alignSliceOpsFcn)
            .setDescription(
                "Aligns slice ops in the correct shape and order required by Keembay");
    }
}

void alignSliceOpsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    auto sliceOps = om.getOps("Slice");

    for(auto op: sliceOps)
    {
        auto opId = op->get<unsigned>("opId");

        std::cout << "Aligning sliceOp " << op->getName() << std::endl;

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
        auto weightSetDimension = outputWidth * outputHeight * outputChannels;

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
        auto outputDataFlows = mv::getOutputDataFlow(om, op);

        auto newOp = om.getSourceOp(newSlice);
        newOp->set<unsigned>("opId", opId);

        mv::setOutputDataFlow(om, newSlice, outputDataFlows);

        newSlice->set<mv::Shape>("OriginalShape", outputShape);
        for(auto& flowPair: outputDataFlows)
        {
            flowPair.first->set<std::array<unsigned short, 2>>("kSize", {outputWidth, outputHeight});
            flowPair.first->set<unsigned>("inputChannels", inputChannels);
            flowPair.first->set<unsigned>("outputChannels", outputChannels);
            flowPair.first->set<mv::QuantizationParams>("quantParams", quantParams);
        }
    }
}
