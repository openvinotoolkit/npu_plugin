#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void ensureDTypeCompatibilityFcn(
    const mv::pass::PassEntry& pass,
    mv::ComputationModel& model,
    mv::TargetDescriptor& targetDesc,
    mv::Element&,
    mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(EnsureDTypeCompatibility)
        .setFunc(ensureDTypeCompatibilityFcn)
        .setDescription(
            "Ensure all HW tasks have compatible dtype combinationsm. "
            "Apply the right mitigations taking into account performance and accuracy"
        );

    }

}

// Most KMB dtype restrictions defined in the HW doc are related to the datatype that's
// resolved at MPE stage.
// Eltwise is different in the way that it bypasses the MPE and it goes directly to PPE
// Even if eltwise uses the same IDU unit dedicated to weights(for the read of the
// second input tensor) it does not inherit any mix datatype support or restrictions from the
// cases that use MPE.
// In short eltwise is forced to have the same dtype for its input tensors.

const std::vector<std::pair<mv::Data::TensorIterator, std::string>>
    dataTypeDiscrepancySolver(
    mv::OpModel &om,
    mv::Data::OpListIterator &opIt,
    mv::DataTypeSupport &dtypeCase)
{

    using tensorSolverFunc = std::function<mv::Data::TensorIterator(
        const mv::Data::OpListIterator &,
        const mv::OpModel &)>;
    const std::unordered_map<std::string, tensorSolverFunc>
    tensorSolverMap =
    {
    {
        "input_0",
        [](const mv::Data::OpListIterator& op,
            const mv::OpModel &) {
            return op->getInputTensor(0);
        }
    },
    {
        "input_1",
        [](const mv::Data::OpListIterator& op,
            const mv::OpModel &opModel) {
            if (op->getOpType() == "Eltwise" ||
                (op->getOpType() == "DPUTask" &&
                op->get<std::string>("taskOp") == "Eltwise"))
                return op->getInputTensor(1);
            return opModel.tensorEnd();
        }
    },
    {
        "weights",
        [](const mv::Data::OpListIterator& op,
            const mv::OpModel &opModel) {
            if (op->hasWeights())
                return op->getInputTensor(1);
            return opModel.tensorEnd();
        }
    },
    {
        "output",
        [](const mv::Data::OpListIterator& op,
            const mv::OpModel &) {
            return op->getOutputTensor(0);
        }
    }
    };

    auto mitigationPlan = std::vector<std::pair<mv::Data::TensorIterator, std::string>>();
    auto isFail = true;
    for (auto failCaseEntry : dtypeCase.failCase)
    {
        auto tensorSolver = tensorSolverMap.find(failCaseEntry.first);
        if (tensorSolver == tensorSolverMap.cend())
            throw mv::RuntimeError(om, opIt->getName() +
                ": No tensor dtype solver registered for dtype " +
                failCaseEntry.first);
        auto tensor = tensorSolver->second(opIt, om);
        if (tensor == om.tensorEnd() ||
            tensor->getDType().toString() != failCaseEntry.second)
            isFail = false;
    }
    if (isFail) {
        for (auto mitigationEntry : dtypeCase.mitigation)
        {
            auto tensor = tensorSolverMap.find(mitigationEntry.first)->second(
                opIt, om);
            if (tensor != om.tensorEnd() &&
                tensor->getDType().toString() != mitigationEntry.second)
                mitigationPlan.push_back({tensor, mitigationEntry.second});
        }
    }
    return mitigationPlan;
}

using dtypeConversionFunc = std::function<void(
    mv::Data::TensorIterator &,
    mv::Data::OpListIterator &,
    mv::OpModel& om)>;

static void ensureDTypeCompatibilityFcn(
    const mv::pass::PassEntry&,
    mv::ComputationModel& model,
    mv::TargetDescriptor& targetDesc,
    mv::Element&,
    mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    // TODO: dtype conversions will behave differently for populated and unpopulated tensors
    // also need to determine of the tensor is input or output of current invetigated op
    // for either input or output flow, placinng a conversion layer will differ.
    const std::unordered_map<std::string, const std::unordered_map<std::string, dtypeConversionFunc>>
        dtypeConversionMap =
        {
        {
            "Float16",
            {
            {
                "UInt8",
                [](const mv::Data::TensorIterator& tensorIt,
                    const mv::Data::OpListIterator& opIt,
                    mv::OpModel&) {
                    throw mv::RuntimeError("DType Compatibility ", opIt->getName() +
                        ": No dtype Float16 -> UInt8 conversion registered for " +
                        "tensor " + tensorIt->getName());
                }
            }
            }
        },
        {
            "UInt8",
            {
            {
                "Float16",
                [](const mv::Data::TensorIterator& tensorIt,
                    const mv::Data::OpListIterator& opIt,
                    mv::OpModel&) {
                    throw mv::RuntimeError("DType Compatibility ", opIt->getName() +
                        ": No dtype UInt8 -> Float16 conversion registered for " +
                        "tensor " + tensorIt->getName());
                }
            }
            }
        }
        };
    auto ops = om.getOps("DPUTask");
    for (auto opIt : ops)
    {
        for (auto dtypeCase : targetDesc.dtypeSupport())
        {
            auto mitigationPlan = dataTypeDiscrepancySolver(om, opIt, dtypeCase);

            for (auto mitigationStep : mitigationPlan)
            {

                auto tensorIt = mitigationStep.first;
                auto targetDtype = mitigationStep.second;

                auto subMapEntry = dtypeConversionMap.find(
                    tensorIt->getDType().toString());

                if (subMapEntry == dtypeConversionMap.cend())
                    throw mv::RuntimeError(om, tensorIt->getName() +
                        ": No dtype conversion registered for source dtype " +
                        tensorIt->getDType().toString());

                auto conversionFunctor = subMapEntry->second.find(
                    targetDtype);

                if (conversionFunctor == subMapEntry->second.cend())
                    throw mv::RuntimeError(om, tensorIt->getName() +
                        ": No dtype conversion registered for target dtype " +
                        tensorIt->getDType().toString() + " from source dtype " +
                        targetDtype);

                std::string errorLog = "Found folowing incompatible dtype case:\n";
                for (auto entry : dtypeCase.failCase) {
                    errorLog += "Tensor to op relation: " + entry.first + "\n";
                    errorLog += "Tensor datatype: " + entry.second + "\n";
                }
                throw mv::RuntimeError("DType Compatibility ", errorLog + "Dtype mitigation for op " +
                    opIt->getName() + " with associated tensor " + tensorIt->getName() +
                    " from " + tensorIt->getDType().toString() + " to " + targetDtype +
                    " is needed!");
            }

        }

    }

}