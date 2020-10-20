#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_input
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args,
            std::string&) -> std::pair<bool, std::size_t>
        {
            if(args.at("dType").get<mv::DType>() == mv::DType("Default"))
                return {false, 1};
            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            outputs.emplace_back(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(),
                    args.at("order").get<mv::Order>());
        };


    }


    namespace op {

        MV_REGISTER_OP(Input)
        .setOutputs({"output"})
        .setArg<mv::Shape>("shape")
        .setArg<mv::DType>("dType")
        .setArg<mv::Order>("order")
        .setOptionalArg<bool>("networkInput", true)
        .setInputCheck(op_input::inputCheckFcn)
        .setOutputDef(op_input::outputDefFcn)
        .setTypeTrait({"exposed", "executable", "optimizable"});

    }

}
