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

void alignTaskWeightsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // Pass main assumption is that we are working on the task graph, no DMA involved yet (just AveragePooling substituted)
    auto dpuTasks = om.getOps("DPUTask");

    for(auto vecIt = dpuTasks.begin(); vecIt != dpuTasks.end(); ++vecIt)
    {
        auto opIt = *vecIt;
        std::string opType = opIt->get<std::string>("taskOp");
        if (opType == "ChannelMajorConvolution" || opType == "DepthwiseConv" || opType == "Conv")
        {
            auto kernel = opIt->getInputTensor(1);
            auto kernelOp = om.getSourceOp(kernel);
            auto kernelShape = kernel->getShape();

            opIt->set<std::array<unsigned short, 2>>("kSize", {kernelShape[0], kernelShape[1]});

            // NOTE: Is the assumption double correct?
            auto weightSetDimension = kernelShape[0]*kernelShape[1]*kernelShape[2];
            mv::Shape newShape({kernelShape[3], 1, 1, mv::round_up(weightSetDimension, 16)});
            std::vector<double> oldData = kernel->getDoubleData();
            oldData.resize(newShape.totalSize(), 0);
            std::vector<double> newData(std::move(oldData));
            auto newKernelOp = om.constant(newData, newShape, kernel->getDType(), kernel->getOrder(), "Aligned"+kernelOp->getName());
            om.removeOp(kernelOp);

            opIt->setInputTensor(newKernelOp, 1, false);
            om.defineFlow(newKernelOp, opIt, 1);
        }
    }
}
