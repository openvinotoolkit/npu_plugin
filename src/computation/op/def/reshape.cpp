#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_reshape
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
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

            mv::Order new_order(inputs[0]->getOrder()); // by default: do not change order
            new_order = args.at("order").get<mv::Order>();

            auto new_shape = args.at("shape").get<mv::Shape>();
            if (new_shape.ndims() != 4)
            {
                new_shape = mv::Shape::augment(new_shape, 4);
            }

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0",  new_shape, dTypeToUse, new_order));
            else
                outputs.push_back(mv::Tensor(":0",  new_shape, dTypeToUse, new_order, args.at("quantParams").get<mv::QuantizationParams>()));

        };

        static std::string empty;

    }

    namespace op {

        // Reshape:
        // Change tensor shape w/o physically moving data
        //
        // By default, tensor's order remains not changed.
        // If you need to change tensor's order as well,
        // please provide a not-empty "order" parameter.
        //
        // For example:
        // Given NxCxHxW input tensor, you may reshape it
        // into Nx(C*H*W) with parameters: order="NC" and
        // shape=mv::Shape({C*H*W, N})

        // TODO: introduce "undefined" value of mv::Order
        // Undefined new order means: do not change order

        MV_REGISTER_OP(Reshape)
        .setInputs({"data0"})
        .setOutputs({"output"})
        .setArg<mv::Shape>("shape")
        .setArg<mv::Order>("order")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_reshape::inputCheckFcn)
        .setOutputDef(op_reshape::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
