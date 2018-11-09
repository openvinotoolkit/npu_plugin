#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            if (inputs[1]->getShape().ndims() != 2)
            {
                errMsg = "Invalid shape of the weights tensor (input 1) - must have a dimensionality of 2, "
                    " has " + std::to_string(inputs[1]->getShape().ndims());
                return {false, 0};
            }

              if (inputs[0]->getShape().totalSize() != inputs[1]->getShape()[0])
            {
                errMsg = "Inconsistent total size of input tensor (input 0) " + std::to_string(inputs[0]->getShape().totalSize()) + 
                    " and 1st dimension of weights tensor (input 1) " + std::to_string(inputs[1]->getShape()[0]);
                return {false, 0};
            }

            return {true, 0};

        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&, std::vector<Tensor>& outputs)
        {

            std::string inputOrder = inputs[0]->getOrder().toString();
            std::string outputOrder = inputOrder.substr(inputOrder.size() - 2);
            outputs.push_back(mv::Tensor(":0", {1, inputs[1]->getShape()[1]}, inputs[0]->getDType(), Order(outputOrder)));
            
        };
    
        MV_REGISTER_OP(FullyConnected)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
