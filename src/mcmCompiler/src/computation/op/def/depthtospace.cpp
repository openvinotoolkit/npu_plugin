#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_depthtospace
    {
        static const std::string defaultMode = "blocks_first";
        static const std::set<std::string> supportedModes = {"blocks_first", "depth_first"};

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            const auto mode = args.at("mode").get<std::string>();
            const auto modesIter = supportedModes.find(mode);
            if (modesIter == supportedModes.end())
            {
                errMsg = "Attempt to set unsupported DepthToSpace mode: " + mode
                       + ". Supported values are:";
                for (const std::string& supportedValue : supportedModes) {
                    errMsg += " ";
                    errMsg += supportedValue;
                }
                return {false, 0};
            }

            if (!inputs.empty() && inputs[0]->getShape().ndims() > 4)
            {
                std::stringstream err;
                err << "Invalid shape of the input tensor (input 0) - must have a dimensionality of 4, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());
                errMsg = err.str();
                return {false, 0};
            }

            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto inputShape = inputs[0]->getShape();
            auto block_size = args.at("block_size").get<uint32_t>();

            size_t W = inputShape[IO_WIDTH_DIMENSION] * block_size;
            size_t H = inputShape[IO_HEIGHT_DIMENSION] * block_size;
            size_t C = inputShape[IO_CHANNEL_DIMENSION] / (block_size * block_size);
            size_t N = inputShape[IO_BATCH_DIMENSION];

            Shape outputShape({W, H, C, N});

            outputs.emplace_back(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder());
        };
    }

    namespace op {
        MV_REGISTER_OP(DepthToSpace)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<std::string>("mode", op_depthtospace::defaultMode)
        .setArg<uint32_t>("block_size")
        .setInputCheck(op_depthtospace::inputCheckFcn)
        .setOutputDef(op_depthtospace::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
