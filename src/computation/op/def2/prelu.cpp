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
            auto inputShape = inputs[0]->getShape();
            auto slopeShape = inputs[1]->getShape();
            
            if (slopeShape.ndims() != 1) 
            {
                errMsg = "Incorrect shape " + slopeShape.toString() + " of slope (must be a vector)";
                return {false, 0};
            }
            
            if (inputShape[-1] != slopeShape[0]) 
            {
                errMsg = "Mismatch in channels dimensions between input (" + std::to_string(inputShape[-1])
                    + ") and slope (" + std::to_string(slopeShape[0]) + ")";
                return {false, 0};
            }


            return {true, 0};

        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&, std::vector<Tensor>& outputs)
        {

            outputs.push_back(mv::Tensor(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder()));

        };
    
        MV_REGISTER_OP(PRelu)
        .setInputs({"data", "slope"})
        .setOutputs({"output"})
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
