#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void alignTaskWeightsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(AlignTaskWeights)
            .setFunc(alignTaskWeightsFcn)
            .setDescription(
                "Aligns weights involved in DPUTasks in the correct shape and order required by Kmb");
    }
}

// This pass aligns weights for Convolutions that will be executed as DPUTasks
// Pass main assumption is that we are working on the task graph, no DMA involved yet
// Another assumption is that if a tensor of weights is involved in more than one OP
// Then either all these ops are DPUTasks or neither of them. Another assumption is
// that all the dpu tasks are the same operation

void alignTaskWeightsFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
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
        bool hasSliceOp = false;
        std::string dpuTaskType;
        for(auto opIt = kernelOp.leftmostChild(); opIt != om.opEnd(); ++opIt)
        {
            if(opIt->getOpType() == "DPUTask")
            {
                hasAtLeastOneDPUTask = true;
                toUpdate.push_back(opIt);
                if(dpuTaskType.empty())
                    dpuTaskType = opIt->get<std::string>("taskOp");
                else if(dpuTaskType != opIt->get<std::string>("taskOp"))
                    throw "Assumption violated!";
            }
            else if (opIt->getOpType() == "Slice")
            {
                hasSliceOp = true;
            }
            else if(hasAtLeastOneDPUTask)
                throw "Assumption violated!";
        }

        if(hasAtLeastOneDPUTask || hasSliceOp)
        {
            auto kernel = kernelOp->getOutputTensor(0);
            auto kernelShape = kernel->getShape();
            auto kernelName = kernelOp->getName();
            auto kernelDType = kernel->getDType();
            mv::QuantizationParams quantParams = {{},{},{},{}};
            if(kernel->hasAttr("quantParams"))
                quantParams = kernel->get<mv::QuantizationParams>("quantParams");

            auto inputChannels = kernelShape[mv::KERNEL_INPUT_CHANNELS];
            auto kernelWidth = kernelShape[mv::KERNEL_WIDTH];
            auto kernelHeight = kernelShape[mv::KERNEL_HEIGHT];
            auto oldWeightLocation = kernel->get<mv::Tensor::MemoryLocation>("Location");

            //Initializions are done assuming regular convolution and then eventually modified for depthwise
            auto outputChannels = kernelShape[mv::KERNEL_OUTPUT_CHANNELS];
            if(dpuTaskType == "DepthwiseConv")
                outputChannels = inputChannels;

            auto weightSetDimension = kernelWidth * kernelHeight * inputChannels;
            if(dpuTaskType == "DepthwiseConv")
                weightSetDimension = kernelWidth * kernelHeight;
            auto weightsDType = kernel->get<mv::DType>("dType");
            auto weightSetDimensionPadded = mv::round_up(weightSetDimension, 16);
            if (weightsDType == mv::DType("Float16"))
                weightSetDimensionPadded = mv::round_up(weightSetDimension, 8);
            auto paddingDifference = weightSetDimensionPadded - weightSetDimension;

            mv::Shape newShape({weightSetDimensionPadded, 1, 1, outputChannels});

            auto oldData = kernel->getData();

            // NOTE: This should be ZeroPoint, not 0
            std::vector<mv::DataElement> newData(newShape.totalSize(), 0);
            unsigned i = 0, j = 0;
            for(unsigned oc = 0; oc < outputChannels; ++oc)
            {
                for(unsigned ws = 0; ws < weightSetDimension; ++ws)
                    newData[j++] = oldData[i++];

                for(unsigned ws = 0; ws < paddingDifference; ++ws)
                    ++j;
            }

            std::string newKernelName = kernelName;
            if(paddingDifference != 0)
                newKernelName = mv::createAlignWeightSetConstantName(kernelName);
            auto outputDataFlows = mv::getOutputDataFlow(om, kernelOp);
            auto newKernel = om.constantDataElement(newData, newShape, kernelDType,
                                                    mv::Order("NHWC"), quantParams, newKernelName);
            auto newKernelOp = om.getSourceOp(newKernel);
            newKernelOp->set<unsigned>("opId", opId);

            if (hasSliceOp)
            {
                //std::cout << "old weights location = " << oldWeightLocation.toString() << std::endl;
                newKernel->set<mv::Tensor::MemoryLocation>("Location", oldWeightLocation);
            }

            mv::setOutputDataFlow(om, newKernel, outputDataFlows);

            newKernel->set<mv::Shape>("OriginalShape", kernelShape);
            for(auto& flowPair: outputDataFlows)
            {
                flowPair.first->set<std::array<unsigned short, 2>>("kSize", {kernelWidth, kernelHeight});
                flowPair.first->set<unsigned>("inputChannels", inputChannels);
                flowPair.first->set<unsigned>("outputChannels", outputChannels);
                flowPair.first->set<mv::QuantizationParams>("quantParams", quantParams);
            }
        }
    }
}
