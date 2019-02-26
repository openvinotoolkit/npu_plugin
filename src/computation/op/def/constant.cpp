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

            DType dtype = args.at("dType").get<mv::DType>();
            if (dtype == DType("Float8") || dtype == DType("Float16") || dtype == DType("Float32") || dtype == DType("Float64"))
            {
                outputs.push_back(mv::Tensor(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(),
                    args.at("order").get<mv::Order>(), args.at("data").get<std::vector<double>>()));
            }
            else
            {
                   outputs.push_back(mv::Tensor(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(),
                    args.at("order").get<mv::Order>(), args.at("data").get<std::vector<int64_t>>()));
            }


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

        MV_REGISTER_OP(Constant__Int)
        .setOutputs({"output"})
        .setArg<std::vector<int64_t>>("data")
        .setArg<mv::Shape>("shape")
        .setArg<mv::DType>("dType")
        .setArg<mv::Order>("order")
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"exposed"});
    }

}
