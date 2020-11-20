#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/tensor/quantization_params.hpp"

namespace mv
{

    namespace op_matmul
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
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& /*args*/, std::vector<Tensor>& outputs)
        {
            const Shape shape = {inputs[0]->getShape()[0], inputs[1]->getShape()[1]};
            outputs.emplace_back(":0", shape, inputs[0]->getDType(), inputs[0]->getOrder());
        };


    }

    namespace op {
        MV_REGISTER_OP(MatMul)
        .setInputs({"data0", "data1"})
        .setOutputs({"output"})
        .setInputCheck(op_matmul::inputCheckFcn)
        .setOutputDef(op_matmul::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
