#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_quantize
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
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            mv::Tensor outputTensor(*inputs[0]);

            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse != mv::DType("Default"))
                outputTensor.setDType(dTypeToUse);

            mv::QuantizationParams quantParams= {{},{},{},{}};

            if (!args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                quantParams = args.at("quantParams").get<mv::QuantizationParams>();

            outputTensor.set<mv::QuantizationParams>("quantParams", quantParams);

            outputs.push_back(std::move(outputTensor));
            outputs[0].setName(outputs[0].getName() + ":0");
            if (outputs[0].hasAttr("flows")) 
                outputs[0].erase("flows");

            if (outputs[0].hasSubTensors())
            {
                for (std::size_t i = 0; i < outputs[0].numSubTensors(); i++)
                {
                    outputs[0].getSubTensor(i).setName(outputs[0].getName() + "sub" + std::to_string(i));
                }
            }
        };
    }

    namespace op
    {

        MV_REGISTER_OP(Quantize)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_quantize::inputCheckFcn)
        .setOutputDef(op_quantize::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
