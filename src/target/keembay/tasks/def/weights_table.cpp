#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_weights_table
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::string&) -> std::pair<bool, std::size_t>
        {

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputIntDefFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
            {
                outputs.push_back(mv::Tensor(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(),
                    args.at("order").get<mv::Order>(), args.at("data").get<std::vector<int64_t>>()));
            }
            else
            {
                outputs.push_back(mv::Tensor(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(),
                    args.at("order").get<mv::Order>(), args.at("data").get<std::vector<int64_t>>(), args.at("quantParams").get<mv::QuantizationParams>()));
            }
        };


    }

    namespace op {
        MV_REGISTER_OP(WeightsTable)
        .setOutputs({"output"})
        .setArg<std::vector<int64_t>>("data")
        .setArg<mv::Shape>("shape")
        .setArg<mv::DType>("dType")
        .setArg<mv::Order>("order")
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{},{},{}))
        .setInputCheck(op_weights_table::inputCheckFcn)
        .setOutputDef(op_weights_table::outputIntDefFcn);
    }

}
