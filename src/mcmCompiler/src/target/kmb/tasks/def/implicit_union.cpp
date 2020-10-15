#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_implicit_union
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto inputShape0 = inputs[0]->getShape();

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

            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto batchAxis = mv::Shape::getAxis("N");

            //NOTE: shape and dtype are not really important here, since implicit union is just to be used to join all outputs into one output node
            // we use the largest shape and dtype - just to mimic the meaning of union

            mv::Shape inputShapeMax = inputs[0]->getShape();

            for (std::size_t i = 1; i < inputs.size(); ++i)
            {
                auto inputShape = inputs[i]->getShape();
                if (inputShape.totalSize() > inputShapeMax.totalSize())
                    inputShapeMax = inputShape;
            }
            inputShapeMax[batchAxis] = inputs.size();

            auto dTypeToUse = inputs[0]->getDType();
            for (std::size_t i = 1; i < inputs.size(); ++i)
            {
                auto inputDType = inputs[i]->getDType();
                if(inputDType != dTypeToUse)
                    if(inputDType.getSizeInBits() < dTypeToUse.getSizeInBits())
                        dTypeToUse = inputDType;
            }

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0", mv::Shape(inputShapeMax), dTypeToUse, inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", mv::Shape(inputShapeMax), dTypeToUse, inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));
        };

    }

    namespace op {


        MV_REGISTER_OP(ImplicitUnion)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setVariableInputNum(true)
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_implicit_union::inputCheckFcn)
        .setOutputDef(op_implicit_union::outputDefFcn);

    }
}
