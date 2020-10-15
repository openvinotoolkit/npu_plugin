#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{
    namespace op_tile
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto input = inputs[0];
            auto inputShape = input->getShape();

            auto axis  = args.at("axis").get<unsigned>();
            auto tiles = args.at("tiles").get<unsigned>();

            if (axis >= inputShape.ndims()) {
                errMsg = "Invalid axis number - has to be less than number of dimensions - 1"
                    + std::to_string(axis);
                return {false, 1};
            }

            if (tiles == 0) {
                errMsg = "Number of tiles shall be more than 0";
                return {false, 2};
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

            mv::Order order(inputs[0]->getOrder());

            auto axis  = args.at("axis").get<unsigned>();
            auto tiles = args.at("tiles").get<unsigned>();

            // calculate output shape from input shape and params
            auto inputShape = inputs[0]->getShape();
            auto ndims = inputShape.ndims();
            mv::Shape outputShape(ndims);
            for (size_t i = 0; i < ndims; i++)
            {
                outputShape[i] = inputShape[i];
                if (i == axis) {
                    outputShape[i] = outputShape[i] * tiles;
                }
            }

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0",  outputShape, dTypeToUse, order));
            else
                outputs.push_back(mv::Tensor(":0",  outputShape, dTypeToUse, order, args.at("quantParams").get<mv::QuantizationParams>()));
        };
    }

    namespace op {

        // Tile layer extends input blob with copies of data along specific axis.

        MV_REGISTER_OP(Tile)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<unsigned>("axis")
        .setArg<unsigned>("tiles")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_tile::inputCheckFcn)
        .setOutputDef(op_tile::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
