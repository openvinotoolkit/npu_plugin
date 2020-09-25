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
        const auto outOrder = args.at("outOrder").get<mv::Order>();
        const auto outShape = args.at("outShape").get<mv::Shape>();

        const auto dType = [&] {
            const auto dType = args.at("dType").get<mv::DType>();
             if (dType == mv::DType("Default")) {
                 return inputs[0]->getDType();
             }
            return dType;
        }();

        const auto quantParams = args.at("quantParams").get<mv::QuantizationParams>();

        if (quantParams.isEmpty()) {
            outputs.emplace_back(":0", outShape, dType, outOrder);
        } else {
            outputs.emplace_back(":0", outShape, dType, outOrder, quantParams);
        }
    };

}

namespace op {

    MV_REGISTER_OP(Custom)
            .setInputs({"inputs"})
            .setOutputs({"output"})
            .setVariableInputNum(true)
            .setArg<std::vector<uint8_t>>("kernelData")
            .setArg<std::vector<uint8_t>>("paramData")
            .setArg<mv::Order>("outOrder")
            .setArg<mv::Shape>("outShape")
            .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
            .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({}, {}, {}, {}))
            .setInputCheck(op_custom::inputCheckFcn)
            .setOutputDef(op_custom::outputDefFcn)
            .setTypeTrait({"executable", "exposed"});

}

}
