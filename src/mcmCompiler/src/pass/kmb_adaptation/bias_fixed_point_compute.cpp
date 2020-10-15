#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void adaptFixedPointComputeFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(AdaptFixedPointCompute)
        .setFunc(adaptFixedPointComputeFcn)
        .setDescription(
            "Adapt constant data for fixed point computation, of format S16.16"
        );

    }

}

using toFloatFunc = std::function<double(const mv::DataElement &)>;

static void adaptFixedPointComputeFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    const std::unordered_map<std::string, toFloatFunc> toFloatConversionMap =
        {
        {
            "Float16",
            [](const mv::DataElement& dataElement) {
                return mv::fp16_to_fp32(static_cast<uint16_t>
                (static_cast<int64_t>(dataElement)));
            }
        },
        {
            "Float32",
            [](const mv::DataElement& dataElement) {
                return static_cast<double>(dataElement);
            }
        },
        {
            "Float64",
            [](const mv::DataElement& dataElement) {
                return static_cast<double>(dataElement);
            }
        },
        {
            "Int32",
            [](const mv::DataElement& dataElement) {
                return static_cast<double>
                (static_cast<int64_t>(dataElement));
            }
        }
        };

    auto floatBiasTensors = std::set<std::string>();
    for (auto& opIt : om.getOps("DPUTask"))
    {
        //NOTE: The order of the hardware is mult_acc->bias->mult/shift
        //So when there is fp16 input there should be s16.16 bias
        auto accDtype = opIt->getInputTensor(0)->getDType();
        if (opIt->hasAttr("bias") && accDtype == mv::DType("Float16") && opIt->hasFloatPrecision())
            floatBiasTensors.insert(opIt->get<std::string>("bias"));
    }

    for (auto biasTensorName : floatBiasTensors){
        auto biasTensor = dm.getTensor(biasTensorName);
        auto conversionFunctor = toFloatConversionMap.find(
            biasTensor->getDType().toString());

        if (conversionFunctor == toFloatConversionMap.cend())
            throw mv::RuntimeError(om, biasTensor->getName() +
                ": No conversion to float registered for bias dtype of " +
                biasTensor->getDType().toString());

        auto biasData = biasTensor->getData();
        auto fixedPointBiasData = std::vector<double>(biasData.size());

        std::transform(
            biasData.cbegin(),
            biasData.cend(),
            fixedPointBiasData.begin(),
            conversionFunctor->second);

        // Regardless if the data is float or integer, we need to convert
        // it to a S16.16 format for the HW compute, which is equivalent
        // to a multiply of 2^16 to align the zero point
        std::transform(
            fixedPointBiasData.begin(),
            fixedPointBiasData.end(),
            fixedPointBiasData.begin(),
            [](double element) {
                return std::round(element * std::pow(2, 16));
            });

        for (size_t idx = 0; idx < biasTensor->size(); idx++)
        {
            biasTensor->at(idx) = fixedPointBiasData[idx];
        }
        biasTensor->setDType(mv::DType("Int32"));
    }

}
