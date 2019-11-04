#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_add
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto inputSize = inputs.size();
            if(inputSize < 2)
            {
                errMsg = "Eltwise needs at least two inputs";
                return {false, 1};
            }

            // NOTE: Compiler assumption. It's very stupid
            // for frontend to give element wise of two populated tensors
            // so we assume that there is one unpopulated tensor and it has
            // to be in position 0
            if(inputs[0]->isPopulated())
            {
                errMsg = "Input 0 of eltwise needs at least two inputs";
                return {false, 1};
            }

            auto input0Shape = inputs[0]->getShape();
            for(std::size_t i = 1; i < inputSize; ++i)
            {
                auto inputIShape = inputs[i]->getShape();
                if ((input0Shape != inputIShape))
                {
                    if(inputIShape.totalSize() != 1 && !inputs[i]->isPopulated())
                    {
                        errMsg = "All the inputs of eltwise ops have to share the same size or the other inputs must have size 1 and be populated";
                        return {false, 1};
                    }
                }
            }
            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&args, std::vector<Tensor>& outputs)
        {
            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0",  inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",  inputs[0]->getShape(), dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));
        };
    }

    namespace op {
        MV_REGISTER_OP(Add)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_add::inputCheckFcn)
        .setOutputDef(op_add::outputDefFcn)
        .setTypeTrait({"executable", "exposed", "optimizable"})
        .setVariableInputNum(true);

    }

}
