#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_constant
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
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
            {
                outputs.push_back(mv::Tensor(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(),
                    args.at("order").get<mv::Order>(), args.at("data").get<std::vector<double>>()));
            }
            else
            {
                outputs.push_back(mv::Tensor(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(),
                    args.at("order").get<mv::Order>(), args.at("data").get<std::vector<double>>(), args.at("quantParams").get<mv::QuantizationParams>()));
            }
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

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDataElementDefFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
            {
                outputs.push_back(mv::Tensor(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(),
                                             args.at("order").get<mv::Order>(),  args.at("data").get<std::vector<mv::DataElement>>()));
            }
            else
            {
                outputs.push_back(mv::Tensor(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(),
                    args.at("order").get<mv::Order>(),  args.at("data").get<std::vector<mv::DataElement>>(),args.at("quantParams").get<mv::QuantizationParams>()));
            }
     };

    }

    namespace op {
        MV_REGISTER_OP(Constant)
        .setOutputs({"output"})
        .setArg<std::vector<double>>("data")
        .setArg<mv::Shape>("shape")
        .setArg<mv::DType>("dType")
        .setArg<mv::Order>("order")
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_constant::inputCheckFcn)
        .setOutputDef(op_constant::outputDefFcn)
        .setTypeTrait({"exposed"});

        MV_REGISTER_OP(ConstantInt)
        .setOutputs({"output"})
        .setArg<std::vector<int64_t>>("data")
        .setArg<mv::Shape>("shape")
        .setArg<mv::DType>("dType")
        .setArg<mv::Order>("order")
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_constant::inputCheckFcn)
        .setOutputDef(op_constant::outputIntDefFcn)
        .setTypeTrait({"exposed"});

        MV_REGISTER_OP(ConstantDataElement)
        .setOutputs({"output"})
        .setArg<std::vector<mv::DataElement>>("data")
        .setArg<mv::Shape>("shape")
        .setArg<mv::DType>("dType")
        .setArg<mv::Order>("order")
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_constant::inputCheckFcn)
        .setOutputDef(op_constant::outputDataElementDefFcn);


    }

}
