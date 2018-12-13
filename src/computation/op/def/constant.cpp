#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op
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
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            outputs.push_back(mv::Tensor(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(), 
                args.at("order").get<mv::Order>(), args.at("data").get<std::vector<double>>()));

        };
    
        MV_REGISTER_OP(Constant)
        .setOutputs({"output"})
        .setArg<std::vector<double>>("data")
        .setArg<mv::Shape>("shape")
        .setArg<mv::DType>("dType")
        .setArg<mv::Order>("order")
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"exposed"});

    }

}
