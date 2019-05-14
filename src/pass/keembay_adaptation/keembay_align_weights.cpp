#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void alignTaskWeightsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AlignTaskWeights)
            .setFunc(alignTaskWeightsFcn)
            .setDescription(
                "Aligns weights involved in DPUTasks in the correct shape and order required by Keembay");
    }
}

// This pass aligns weights for Convolutions that will be executed as DPUTasks
// Pass main assumption is that we are working on the task graph, no DMA involved yet
// Another assumption is that if a tensor of weights is involved in more than one OP
// Then either all these ops are DPUTasks or neither of them.

void alignTaskWeightsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    auto constants = om.getOps("Constant");
    auto constantInts = om.getOps("ConstantInt");
    auto constantsDataElements = om.getOps("ConstantDataElement");

    constants.insert(constants.end(), std::make_move_iterator(constantInts.begin()), std::make_move_iterator(constantInts.end()));
    constants.insert(constants.end(), std::make_move_iterator(constantsDataElements.begin()), std::make_move_iterator(constantsDataElements.end()));

    for(auto vecIt = constants.begin(); vecIt != constants.end(); ++vecIt)
    {
        auto kernelOp = *vecIt;
        auto opId = kernelOp->get<unsigned>("opId");

        std::vector<mv::Data::OpListIterator> toUpdate;
        bool hasAtLeastOneDPUTask = false;
        for(auto opIt = kernelOp.leftmostChild(); opIt != om.opEnd(); ++opIt)
        {
            if(opIt->getOpType() == "DPUTask")
            {
                hasAtLeastOneDPUTask = true;
                toUpdate.push_back(opIt);
            }
            else if(hasAtLeastOneDPUTask)
                throw "Assumption violated!";
        }

        if(hasAtLeastOneDPUTask)
        {
            auto kernel = kernelOp->getOutputTensor(0);
            auto opIt = kernelOp.leftmostChild();
            auto kernelShape = kernel->getShape();
            mv::QuantizationParams quantParams = {{},{},{},{}};
            if(kernel->hasAttr("quantParams"))
                quantParams = kernel->get<mv::QuantizationParams>("quantParams");
            auto weightSetDimension = kernelShape[mv::KERNEL_WIDTH]*kernelShape[mv::KERNEL_HEIGHT]*kernelShape[mv::KERNEL_INPUT_CHANNELS];
            auto weightSetDimensionPadded = mv::round_up(weightSetDimension, 16);
            auto paddingDifference = weightSetDimensionPadded - weightSetDimension;
            mv::Shape newShape({weightSetDimensionPadded, 1, 1, kernelShape[mv::KERNEL_OUTPUT_CHANNELS]});

            //NOTE: This three lines have to be corrected
            auto oldData = kernel->getData();

            std::vector<mv::DataElement> newData(newShape.totalSize(), 0);
            unsigned i = 0, j = 0;
            for(unsigned oc = 0; oc < kernelShape[mv::KERNEL_OUTPUT_CHANNELS]; ++oc)
            {
                for(unsigned ws = 0; ws < weightSetDimension; ++ws)
                    newData[j++] = oldData[i++];

                for(unsigned ws = 0; ws < paddingDifference; ++ws)
                    ++j;
            }

            auto newKernel = om.constantDataElement(newData, newShape, kernel->getDType(), mv::Order("NHWC"), quantParams,"Aligned"+kernelOp->getName());

            om.getSourceOp(newKernel)->set<unsigned>("opId", opId);
            om.removeOp(kernelOp);
            for(auto toUpdateIt = toUpdate.begin(); toUpdateIt != toUpdate.end(); ++toUpdateIt)
            {
                (*toUpdateIt)->set<std::array<unsigned short, 2>>("kSize", {kernelShape[mv::KERNEL_WIDTH], kernelShape[mv::KERNEL_HEIGHT]});
                (*toUpdateIt)->set<unsigned>("inputChannels", kernelShape[mv::KERNEL_INPUT_CHANNELS]);
                (*toUpdateIt)->setInputTensor(newKernel, 1, false);
                (*toUpdateIt)->set<mv::QuantizationParams>("quantParams", quantParams);
                om.defineFlow(newKernel, (*toUpdateIt), 1);
            }
        }
    }
}
