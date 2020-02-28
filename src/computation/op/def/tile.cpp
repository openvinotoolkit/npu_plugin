#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{
    namespace op_tile
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::string&) -> std::pair<bool, std::size_t>
        {
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

            // auto axis  = args.at("axis").get<size_t>();
            // auto tiles = args.at("tiles").get<size_t>();
            // TODO: calculate output shape from input shape and params

            auto new_shape = args.at("output_shape").get<mv::Shape>();;

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0",  new_shape, dTypeToUse, order));
            else
                outputs.push_back(mv::Tensor(":0",  new_shape, dTypeToUse, order, args.at("quantParams").get<mv::QuantizationParams>()));
        };
    }

    namespace op {

        // Tile layer extends input blob with copies of data along specific axis.

        MV_REGISTER_OP(Tile)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<std::string>("axis")
        .setArg<std::string>("tiles")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_tile::inputCheckFcn)
        .setOutputDef(op_tile::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}
