#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_flatten
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto axis = args.at("axis").get<int64_t>();

            // Verify valid axis param
            if ((axis < 0) || (axis > 3))
            {
                std::stringstream err;
                err << "only axis values from 0 to 3 are supported; axis = " << axis;
                errMsg = err.str();
                return {false, 0};
            }

            // Verify valid end_axis param
            auto end_axis = args.at("end_axis").get<int64_t>();
            if ((end_axis < -3) || (end_axis > 3))
            {
                std::stringstream err;
                err << "only end_axis values from -3 to 3 are supported; end_axis = " << end_axis;
                errMsg = err.str();
                return {false, 0};
            }
            if (end_axis < 0)
            {
                end_axis += 4;
            }
            if (end_axis < axis)
            {
                std::stringstream err;
                err << "end_axis must be equal-to or right-of axis; axis=" << axis << "; end_axis=" << end_axis;
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

            auto axis = args.at("axis").get<int64_t>();
            auto end_axis = args.at("end_axis").get<int64_t>();
            if (end_axis < 0)
            {
                end_axis += 4;
            }

            // Calculate new flattened dim value
            auto inputShape = inputs[0]->getShape();
            auto input_dims = inputShape.ndims();
            auto flattened_dim_value = 1;
            for (auto i=axis; i<=end_axis; i++)
            {
                flattened_dim_value *= inputShape[input_dims-1 - i];
            }

            // Build new outputShape
            auto new_dims = input_dims - (end_axis - axis);
            auto outputShape = mv::Shape(new_dims);
            // Beginning dims
            for (auto i=0; i<axis; i++)
            {
                outputShape[new_dims-1 - i] = inputShape[input_dims-1 - i];
            }
            // Flattened dim (if necessary)
            if (end_axis > axis)
                outputShape[new_dims-1 - (axis)] = flattened_dim_value;
            // Remaining dims
            auto num_flattened_dims = (end_axis==axis) ? 0 : 1;
            for (auto i=0; i<input_dims-(end_axis+num_flattened_dims); i++)
            {
                outputShape[new_dims-1 - (axis+num_flattened_dims) - i] = inputShape[input_dims-1 - (end_axis) - i];
            }

            // Expand new shape to 4D
            outputShape = mv::Shape::augment_major(outputShape, 4);

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0",  outputShape, dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",  outputShape, dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));

        };

        static std::string empty;

    }

    namespace op {

        MV_REGISTER_OP(Flatten)
        .setInputs({"input"})
        .setOutputs({"output"})
        .setOptionalArg<int64_t>("axis", 1)
        .setOptionalArg<int64_t>("end_axis", 3)
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_flatten::inputCheckFcn)
        .setOutputDef(op_flatten::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
