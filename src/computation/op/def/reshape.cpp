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
            mv::Order new_order(inputs[0]->getOrder()); // by default: do not change order

            auto order_str = args.at("order").get<std::string>();
            if (!order_str.empty())
                new_order = mv::Order(order_str);

            outputs.push_back(mv::Tensor(":0",  args.at("shape").get<mv::Shape>(), inputs[0]->getDType(), new_order));
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
        .setOptionalArg<std::string>("order", op_reshape::empty)
        .setInputCheck(op_reshape::inputCheckFcn)
        .setOutputDef(op_reshape::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
