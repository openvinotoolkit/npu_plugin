#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_reorder
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto input = inputs[0];
            if (inputs.size() != 1)
            {
                std::stringstream err;
                err << "Too many inputs (must be 1): " << inputs.size();
                errMsg = err.str();
                return {false, 0};
            }

            // New order must be a permutation of old order:
            auto old_order = input->getOrder();
            auto new_order = args.at("order").get<mv::Order>();
            auto old_order_str = old_order.toString();
            auto new_order_str = new_order.toString();
            std::sort(old_order_str.begin(), old_order_str.end());
            std::sort(new_order_str.begin(), new_order_str.end());
            if (old_order_str != new_order_str)
            {
                std::stringstream err;
                err << "Incompatible orders: old order=" << old_order.toString()
                                       << ", new order=" << new_order.toString();
                errMsg = err.str();
                return {false, 0};
            }

            return {true, 0};

        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            auto input = inputs[0];
            auto outputShape = input->getShape();
            auto outputOrder = args.at("order").get<mv::Order>();

            outputs.push_back(mv::Tensor(":0", outputShape, input->getDType(), outputOrder));
        
        };


    }

    namespace op {
        // Reorder:
        // Change tensor's order without touching its shape
        // and without physically transposing tensor's data.
        // New shape must be a permutation of old the shape.

        MV_REGISTER_OP(Reorder)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::Order>("order")
        .setInputCheck(op_reorder::inputCheckFcn)
        .setOutputDef(op_reorder::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
