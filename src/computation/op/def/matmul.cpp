#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

namespace mv
{

    namespace op
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            if (inputs[0]->getShape().ndims() != 2)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 2, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());
                return {false, 1};
            }

            if (inputs[1]->getShape().ndims() != 2)
            {
                errMsg = "Invalid shape of the parameters tensor (input 1) - must have a dimensionality of 2, "
                    " has " + std::to_string(inputs[1]->getShape().ndims());
                return {false, 1};
            }

            if (inputs[0]->getShape()[1] != inputs[1]->getShape()[0]) 
            {
                errMsg =  "Mismatch between the second dimensinon of the input tensor (input 0) " + std::to_string(inputs[0]->getShape()[1]) +
                " and the first dimension of the parameters tensor (input 1) " + std::to_string(inputs[1]->getShape()[0]);
                return {false, 1};
            }

            return {true, 0};

        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty() == true)
                outputs.push_back(mv::Tensor(":0", {inputs[0]->getShape()[0], inputs[1]->getShape()[1]}, inputs[0]->getDType(), inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", {inputs[0]->getShape()[0], inputs[1]->getShape()[1]}, inputs[0]->getDType(), inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));
        };

        MV_REGISTER_OP(MatMul)
        .setInputs({"data0", "data1"})
        .setOutputs({"output"})
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
