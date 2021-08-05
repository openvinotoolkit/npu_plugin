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
        const auto dType = opIt->getOutputTensor(0)->getDType();
        const auto attrs = opIt->getOutputTensor(0)->getAttrs({"flows", "Location"});
        const auto sourceTensorName = sourceTensor->getName() + "_copy";
        const auto sourceTensorDType = sourceTensor->getDType();
        
        auto sourceTensorCopy = om.copy(sourceTensorName, sourceTensor);
        om.getOp(sourceTensorName)->set<unsigned>("opId", opIt->get<unsigned>("opId"));
        sourceTensorCopy->setDType(sourceTensorDType);

        if (sourceTensor->isQuantized())
            sourceTensorCopy->setQuantParams(sourceTensor->getQuantParams());

        const auto eltwiseOpName = opIt->getName() + "_Eltwise";
        auto eltwiseOp = om.eltwise(eltwiseOpName, { sourceTensor , sourceTensorCopy }, "And");
        eltwiseOp->setAttrs(attrs);
        eltwiseOp->setDType(dType);
        eltwiseOp->setQuantParams(outputTensorQuantizationParams);
        om.getOp(eltwiseOpName)->set<unsigned>("opId", opIt->get<unsigned>("opId"));

        mv::linkNewOperationsReplacement(parentOpIt, eltwiseOp, om, opIt);
    }
}
