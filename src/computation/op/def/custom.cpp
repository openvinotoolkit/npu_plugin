#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

namespace op_custom
{
    static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
        const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
        [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
        std::string&) -> std::pair<bool, std::size_t>
    {
        return {true, 0};
    };

    static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
        std::vector<Tensor>&)> outputDefFcn =
        [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
           std::vector<Tensor>& outputs)
    {
        // dType argument is ignored (output data type is stored in tensorInfo)
        const auto outputsInfo = args.at("outputsInfo").get<std::vector<mv::TensorInfo>>();
        const auto quantParams = args.at("quantParams").get<mv::QuantizationParams>();

        for (size_t i = 0; i < outputsInfo.size(); i++) {
            auto dType = outputsInfo[i].type();
            if (dType == mv::DType("Default")) {
                dType = inputs[0]->getDType();
            }

            if (quantParams.isEmpty()) {
                outputs.emplace_back(":" + std::to_string(i), outputsInfo[i].shape(), dType,
                                     outputsInfo[i].order());
            } else {
                outputs.emplace_back(":" + std::to_string(i), outputsInfo[i].shape(), dType,
                                     outputsInfo[i].order(), quantParams);
            }
        }
    };

}

namespace op {

    MV_REGISTER_OP(Custom)
            .setInputs({"inputs"})
            .setOutputs({"outputs"})
            .setVariableInputNum(true)
            .setArg<std::vector<uint8_t>>("kernelData")
            .setArg<std::vector<uint8_t>>("paramData")
            .setArg<std::vector<mv::TensorInfo>>("outputsInfo")
            .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
            .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({}, {}, {}, {}))
            .setInputCheck(op_custom::inputCheckFcn)
            .setOutputDef(op_custom::outputDefFcn)
            .setTypeTrait({"executable", "exposed"});

}

}
