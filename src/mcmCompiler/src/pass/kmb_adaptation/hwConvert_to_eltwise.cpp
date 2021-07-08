#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void hwConvertToEltwiseFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(HwConvertToEltwise)
        .setFunc(hwConvertToEltwiseFcn)
        .setDescription(
            "Change HwConvert operations with Eltwise operation"
        );
    }
}

void hwConvertToEltwiseFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto ops = om.getOps("HwConvert");
    for (auto& opIt : ops)
    {
        mv::QuantizationParams outputTensorQuantizationParams = {{},{},{},{}};
        if (opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT)->isQuantized())
        {
            outputTensorQuantizationParams  = opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT)->getQuantParams();
        }
        auto sourceTensor = opIt->getInputTensor(mv::IO_TENSOR_INPUT);
        auto parentOpIt = om.getSourceOp(sourceTensor);
        const auto attrs = opIt->getAttrs({"flows", "Location"});

        auto sourceTensorCopy = om.copy(sourceTensor->getName() + "_1",sourceTensor);
        if (sourceTensor->isQuantized())
            sourceTensorCopy->setQuantParams(sourceTensor->getQuantParams());

        auto eltwiseOp = om.eltwise(opIt->getName() + "_Eltwise", { sourceTensor , sourceTensorCopy }, "And");
        eltwiseOp->setAttrs(attrs);
        eltwiseOp->setQuantParams(outputTensorQuantizationParams);

        mv::linkNewOperationsReplacement(parentOpIt, eltwiseOp, om, opIt);
    }
}
