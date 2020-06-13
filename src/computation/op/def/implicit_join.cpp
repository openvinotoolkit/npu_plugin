#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_implicit_join
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto inputShape0 = inputs[0]->getShape();
            // Axis be either "N", "C", "D", "H", or "W"
            auto axisToConcat = args.at("axis").get<std::string>();
            std::vector<std::size_t> numericAxisToConcat;

            for (size_t i = 0; i < axisToConcat.size(); i++)
                numericAxisToConcat.push_back(mv::Shape::getAxis(axisToConcat.substr(i,1)));


            if (inputShape0.ndims() != 4)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 4, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());

                return {false, 0};
            }

            for (std::size_t i = 1; i < inputs.size(); ++i)
            {
                auto inputShapeI = inputs[i]->getShape();

                if (inputShapeI.ndims() != 4)
                {
                    errMsg = "Invalid shape of the input tensor (input " + std::to_string(i) + ") - must have a dimensionality of 4, "
                        " has " + std::to_string(inputShapeI.ndims());
                    return {false, 0};
                }

                // Based on concat axis, the other dimensions should match
                for(std::size_t shapeDimension = 0; shapeDimension < inputShape0.ndims(); ++shapeDimension)
                {
                    bool axisIsForConcat = false;
                    for(size_t j = 0; j < numericAxisToConcat.size(); j++)
                        if(shapeDimension == numericAxisToConcat[j])
                        {
                            axisIsForConcat = true;
                            break;
                        }

                    if (axisIsForConcat)
                        continue;
                    if (inputShapeI[shapeDimension] != inputShape0[shapeDimension])
                    {
                        std::ostringstream strm;
                        strm
                                << "Invalid shape of the input " << i << " tensor "
                                << "(" << shapeDimension << ":" << inputShapeI[shapeDimension]
                                << " - inconsistent with the dimension of the first input "
                                << "(" << inputShape0[shapeDimension] << ")";

                        errMsg = strm.str();

                        return {false, 0};
                    }
                }
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            // Axis be either "N", "C", "D", "H", or "W"
            auto axisToConcat = args.at("axis").get<std::string>();
            std::vector<std::size_t> numericAxisToConcat;

            for (size_t i = 0; i < axisToConcat.size(); i++)
                numericAxisToConcat.push_back(mv::Shape::getAxis(axisToConcat.substr(i,1)));

            std::vector<std::size_t> inputShape0(inputs[0]->getShape());

            for (std::size_t i = 1; i < inputs.size(); ++i)
            {
                auto inputShape = inputs[i]->getShape();
                for(size_t axis = 0; axis < numericAxisToConcat.size(); axis++)
                    inputShape0[numericAxisToConcat[axis]] += inputShape[numericAxisToConcat[axis]];

            }
            for(size_t axis = 0; axis < numericAxisToConcat.size(); axis++)
                inputShape0[numericAxisToConcat[axis]] /= numericAxisToConcat.size();

            //NOTE/ASSUMPTION: If input DTypes are different, we concatenate with smallest DType.
            auto dTypeToUse = inputs[0]->getDType();
            for (std::size_t i = 1; i < inputs.size(); ++i)
            {
                auto inputDType = inputs[i]->getDType();
                if(inputDType != dTypeToUse)
                    if(inputDType.getSizeInBits() < dTypeToUse.getSizeInBits())
                        dTypeToUse = inputDType;
            }

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", mv::Shape(inputShape0), dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", mv::Shape(inputShape0), dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));
        };

        // Default axis is channels (like for Intel Inference Engine)
        static std::string defaultChannels = "HW";
    }

    namespace op {


        MV_REGISTER_OP(ImplicitJoin)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setVariableInputNum(true)
        .setOptionalArg<std::string>("axis", op_implicit_join::defaultChannels)
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_implicit_join::inputCheckFcn)
        .setOutputDef(op_implicit_join::outputDefFcn);

    }
}
