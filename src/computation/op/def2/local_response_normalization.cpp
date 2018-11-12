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
              if (inputs[0]->getShape().ndims() != 3)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 3, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());
                
                return {false, 0};
            }

            return {true, 0};

        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            outputs.push_back(mv::Tensor(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(), 
                args.at("order").get<mv::Order>(), args.at("data").get<std::vector<double>>()));

        };
        
        //NOTE: Myriad X only can recieve bias and size parameters for LRN
        MV_REGISTER_OP(LocalResponseNormalization)
        .setOutputs({"output"})
        .setArg<std::vector<double>>("data")
        .setArg<mv::Order>("bias")
        .setArg<mv::Order>("size")
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"exposed"});

    }

}
