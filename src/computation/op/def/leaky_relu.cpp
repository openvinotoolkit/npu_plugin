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
            if (inputs[0]->getShape().ndims() != 3)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 3, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());
                
                return {false, 0};
            }
            
            auto alpha = args.at("alpha").get<unsigned>();

            if (alpha < 0)
            {
                errMsg = "Invalid value for alpha parameter - must be a positive numnber";
                    
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
    
        MV_REGISTER_OP(LeakyRelu)
        .setInputs({"data"})
        .setOptionalArg<unsigned>("alpha", 1)
        .setOutputs({"output"})
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed", "automatic_api"});

    }

}
