#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_padding_concat
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& /*args*/,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            // Padding always on "W" axis
            auto numericAxisToConcat = mv::IO_WIDTH_DIMENSION;

            if(inputs.size() != 2)
            {
                errMsg = "Invalid number of inputs to Padding Concat, 2 expected, "
                    " has " + std::to_string(inputs.size() + 1);

                return {false, 0};
            }

            auto inputShape = inputs[0]->getShape();
            auto paddingShape = inputs[1]->getShape();

            if (inputShape.ndims() != 4)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 4, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());

                return {false, 0};
            }

            if (paddingShape.ndims() != 4)
            {
                errMsg = "Invalid shape of the input tensor (input 1) - must have a dimensionality of 4, "
                    " has " + std::to_string(paddingShape.ndims());
                return {false, 0};
            }

            if (paddingShape[mv::IO_WIDTH_DIMENSION] % 16 != 0)
            {
                errMsg = "Invalid padding width of the input tensor (input 1) - must be a multiple of 16, "
                    " has " + std::to_string(paddingShape[mv::IO_WIDTH_DIMENSION]);
                return {false, 0};
            }

            if (paddingShape[mv::IO_WIDTH_DIMENSION] <= inputShape[mv::IO_WIDTH_DIMENSION])
            {
                errMsg = "Invalid padding width of the input tensor (input 1) - must be greater then input width, "
                    " input 1 " + std::to_string(paddingShape[mv::IO_WIDTH_DIMENSION]) + ", input 0 " 
                    + std::to_string(inputShape[mv::IO_WIDTH_DIMENSION]);
                return {false, 0};
            }

            if (inputs[0]->getDType().toString() != inputs[1]->getDType().toString())
            {
                errMsg = "Invalid DType of the input tensor (input 1) - must be equalt to input tenosr 0, "
                    " which has " + inputs[0]->getDType().toString();
                return {false, 0};
            }

            // Based on concat axis, the other dimensions should match
            for(std::size_t shapeDimension = 0; shapeDimension < inputShape.ndims(); ++shapeDimension)
            {
                if(shapeDimension == numericAxisToConcat)
                    continue;
                if (paddingShape[shapeDimension] != inputShape[shapeDimension])
                {
                    std::ostringstream strm;
                    strm
                            << "Invalid shape of the input 1 tensor "
                            << "(" << shapeDimension << ":" << paddingShape[shapeDimension]
                            << " - inconsistent with the dimension of the first input "
                            << "(" << inputShape[shapeDimension] << ")";

                    errMsg = strm.str();

                    return {false, 0};
                }
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& /*args*/, std::vector<Tensor>& outputs)
        {
            // Output shape equal to the padding tensor shape 
            std::vector<std::size_t> outputShape(inputs[1]->getShape());

            outputs.emplace_back(":0", mv::Shape(outputShape), inputs[0]->getDType(), inputs[0]->getOrder());
        };

        // Default axis is channels (like for Intel Inference Engine)
        static std::string channels = "W";
    }

    namespace op {


        MV_REGISTER_OP(PaddingConcat)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setVariableInputNum(true)
        .setInputCheck(op_padding_concat::inputCheckFcn)
        .setOutputDef(op_padding_concat::outputDefFcn);

    }
}
