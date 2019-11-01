#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_argmax
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            // Check for valid axis value
            auto axis = args.at("axis").get<int64_t>();
            if ((!(axis >= -3) && (axis <= 3)) && (axis != 99))
            {
                std::stringstream err;
                err << "Invalid axis value (must be -3 to 3, or 99 for no axis): " << axis;
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

            // Axis is based off NCHW channel ordering, i.e. 0 = N, W=3
            // This is consistent with the TensorReference ordering of dimensions
            // 99 to signal nothing specified, since -3..3 including 0 are valid values
            auto outputShape = inputs[0]->getShape();
            auto axis = args.at("axis").get<int64_t>();

            // Handle negative axis
            if (axis < 0)
            {
                axis = 4 + axis;
            }

            // Modify outputShape based on axis
            if (axis == 99)
            {
                outputShape[3] = 1;
                outputShape[2] = 1;
                outputShape[1] = 1;
                outputShape[0] = 1;
            }
            else if (axis == 0)
            {
                outputShape[3] = 1;
                outputShape[2] = 1;
                outputShape[1] = 1;
            }
            else if (axis == 1)
            {
                outputShape[3] = 1;
                outputShape[2] = 1;
            }
            else if (axis == 2)
            {
                outputShape[3] = 1;
            }

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0",  outputShape, dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",  outputShape, dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));
        };

    }

    namespace op {

        MV_REGISTER_OP(Argmax)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<int64_t>("out_max_val")
        .setArg<int64_t>("top_k")
        .setOptionalArg<int64_t>("axis", 99)
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_argmax::inputCheckFcn)
        .setOutputDef(op_argmax::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
