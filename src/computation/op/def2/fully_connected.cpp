#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto inputShape0 = inputs[0]->getShape();
            auto inputShape1 = inputs[1]->getShape();
            

            if (inputShape1.ndims() != 2)
            {
                errMsg = "Invalid shape of the weights tensor (input 1) - must have a dimensionality of 2, "
                    " has " + std::to_string(inputShape1.ndims());
                return {false, 0};
            }

              if (inputShape0.totalSize() != inputShape1[0])
            {
                errMsg = "Inconsistent total size of input tensor (input 0) " + std::to_string(inputShape0.totalSize()) + 
                    " and 1st dimension of weights tensor (input 1) " + std::to_string(inputShape1[0]);
                return {false, 0};
            }

            return {true, 0};

        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto input1Shape = inputs[1]->getShape();

            outputs.push_back(mv::Tensor(":0", {1, input1Shape[1]}, inputs[0]->getDType(), inputs[0]->getOrder()));

        };
    
        MV_REGISTER_OP(FullyConnected)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
