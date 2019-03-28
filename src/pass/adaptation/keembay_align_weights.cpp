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

void alignTaskWeightsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
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
        bool hasOneDPUTask = false;
        for(auto opIt = kernelOp.leftmostChild(); opIt != om.opEnd(); ++opIt)
        {
            if(opIt->getOpType() == "DPUTask")
            {
                hasOneDPUTask = true;
                toUpdate.push_back(opIt);
            }
            else if(!hasOneDPUTask)
                throw "Assumption violated!";
        }

        if(hasOneDPUTask)
        {
            auto kernel = kernelOp->getOutputTensor(0);
            auto kernelShape = kernel->getShape();

            auto weightSetDimension = kernelShape[0]*kernelShape[1]*kernelShape[2];
            mv::Shape newShape({kernelShape[3], 1, 1, mv::round_up(weightSetDimension, 16)});
            auto oldData = kernel->getData();
            oldData.resize(newShape.totalSize(), 0);
            auto newData(std::move(oldData));
            auto newKernel = om.constantDataElement(newData, newShape, kernel->getDType(), kernel->getOrder(), "Aligned"+kernelOp->getName());
            om.getSourceOp(newKernel)->set<unsigned>("opId", opId);
            om.removeOp(kernelOp);
            for(auto toUpdateIt = toUpdate.begin(); toUpdateIt != toUpdate.end(); ++toUpdateIt)
            {
                (*toUpdateIt)->set<std::array<unsigned short, 2>>("kSize", {kernelShape[0], kernelShape[1]});
                (*toUpdateIt)->set<unsigned>("inputChannels", kernelShape[2]);
                (*toUpdateIt)->setInputTensor(newKernel, 1, false);
                om.defineFlow(newKernel, (*toUpdateIt), 1);
            }
        }
    }
}
