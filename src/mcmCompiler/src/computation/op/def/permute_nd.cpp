#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_permute_nd
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
            auto new_order = args.at("perm_order").get<std::vector<int64_t>>();
            // TODO sanity checks
            auto old_order_str = old_order.toString();
            if (old_order_str.size() != new_order.size()) {
                std::stringstream err;
                err << "Incompatible orders: old order=" << old_order_str
                                       << ", new order={ ";
                for (const auto& dim : new_order) {
                    err << dim << " ";
                }
                err << "}";
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
            auto outputOrder = input->getOrder();
            auto old_order = input->getOrder();
            auto new_order = args.at("perm_order").get<std::vector<int64_t>>();
            auto old_order_str = old_order.toString();

            auto inputShape = input->getShape();
            auto ndims = inputShape.ndims();
            mv::Shape outputShape(ndims);

            for (size_t i = 0; i < ndims; i++)
            {
                outputShape[i] = inputShape[ndims - 1 - new_order.at(ndims - i - 1)];
            }

            // output tensor uses permuted shape with old order
            outputs.emplace_back(":0", outputShape, input->getDType(), old_order);
        };

    }



    namespace op {
        // N-dimensional permutation:
        MV_REGISTER_OP(PermuteND)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<std::vector<int64_t>>("perm_order")
        .setInputCheck(op_permute_nd::inputCheckFcn)
        .setOutputDef(op_permute_nd::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
