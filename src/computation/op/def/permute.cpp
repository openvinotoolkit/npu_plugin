#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_permute
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

            auto dTypeToUse = args.at("dType").get<mv::DType>();
            if(dTypeToUse == mv::DType("Default"))
                dTypeToUse = inputs[0]->getDType();

            auto input = inputs[0];
            auto outputOrder = input->getOrder();

            // Permute tensor Shape according to new Order
            auto old_order = input->getOrder();
            auto new_order = args.at("order").get<mv::Order>();
            auto old_order_str = old_order.toString();
            auto new_order_str = new_order.toString();
            auto inputShape = input->getShape();
            auto ndims = inputShape.ndims();
            mv::Shape outputShape(ndims);
            for (size_t i=0; i < ndims; i++)
            {
                auto j = old_order_str.find(new_order_str[i]);
                assert(j != std::string::npos && j < ndims);
                outputShape[(ndims - 1) - i] = inputShape[(ndims - 1) - j]; // NB: inverse enumeration of dimensions
            }

            outputs.push_back(mv::Tensor(":0", outputShape, dTypeToUse, outputOrder));
        
        };

    }



    namespace op {
        // Permute:
        // Physically transpose tensor's data according to
        // the given permutation of its dimensions.
        // For example, given "NCHW" tensor of 8x3x320x200,
        // permuting last two coordinates like "NCWH" will
        // make it shaped as 8x3x200x320, but tensor order
        // would still remain "NCHW".

        // NOTE: Is this type of operation really necessary in our compiler
        // given our shape/order assumption?

        MV_REGISTER_OP(Permute)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::Order>("order")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_permute::inputCheckFcn)
        .setOutputDef(op_permute::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
