#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_implicit_reshape
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
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();

            mv::Order order(inputs[0]->getOrder()); // by default: do not change order

            auto new_shape = args.at("shape").get<mv::Shape>();
            if (new_shape.ndims() != 4)
            {
                new_shape = mv::Shape::augment(new_shape, 4);
            }

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", new_shape,  dTypeToUse, order));
            else
                outputs.push_back(mv::Tensor(":0", new_shape,  dTypeToUse, order, args.at("quantParams").get<mv::QuantizationParams>()));

        };

        static std::string empty;

    }

    namespace op {


        MV_REGISTER_OP(ImplicitReshape)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        //.setVariableInputNum(true)
        //.setOptionalArg<std::string>("axis", op_implicit_reshape::channels)
        .setArg<mv::Shape>("shape")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_implicit_reshape::inputCheckFcn)
        .setOutputDef(op_implicit_reshape::outputDefFcn);

    }
}
