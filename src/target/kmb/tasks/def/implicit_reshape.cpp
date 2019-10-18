#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_implicit_reshape
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto order_str = args.at("order").get<std::string>();
            if (!order_str.empty())
            {
                try
                {
                    mv::Order order(order_str);
                }
                catch(...)
                {
                    errMsg = "Invalid parameter: order=" + order_str;
                    return {false, 1};
                }
            }

            if (inputs[0]->getShape().totalSize() != args.at("shape").get<mv::Shape>().totalSize())
            {
                errMsg = "Invalid conversino of the original shape " + inputs[0]->getShape().toString() + " and the output shape "
                + args.at("shape").get<mv::Shape>().toString() + " - must have equal total number of elements";
                return {false, 1};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();

            mv::Order new_order(inputs[0]->getOrder()); // by default: do not change order

            auto order_str = args.at("order").get<std::string>();
            if (!order_str.empty())
                new_order = mv::Order(order_str);

            auto new_shape = args.at("shape").get<mv::Shape>();
            if (new_shape.ndims() != 4)
            {
                new_shape = mv::Shape::augment(new_shape, 4);
            }
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", new_shape,  dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", new_shape,  dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));

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
        .setOptionalArg<std::string>("order", op_reshape::empty)
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_implicit_reshape::inputCheckFcn)
        .setOutputDef(op_implicit_reshape::outputDefFcn);

    }
}
