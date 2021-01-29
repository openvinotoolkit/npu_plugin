#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/tensor/tiling.hpp"

namespace mv
{

    namespace op_pad
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
                                                          const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
                [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
                   std::string& errMsg) -> std::pair<bool, std::size_t>
                {
                    if (inputs[0]->size() == 0)
                    {
                        errMsg = "Input tensor is empty";
                        return {false, 0};
                    }
                    if (inputs[0]->getShape().ndims() != 4)
                    {
                        errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 4, "
                                 " has " + std::to_string(inputs[0]->getShape().ndims());
                        return {false, 0};
                    }

                    return {true, 0};
                };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
                                  std::vector<Tensor>&)> outputDefFcn =
                [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
                {
                    auto inputShape = inputs[0]->getShape();

                    auto pads_begin = args.at("pads_begin").get<std::array<unsigned short, 4>>();
                    auto pads_end = args.at("pads_end").get<std::array<unsigned short, 4>>();

                    //use the 'memory ordered' for pad's values in according to real order - NCHW
                    size_t N = inputShape[IO_BATCH_DIMENSION]   + pads_begin[0] + pads_end[0];
                    size_t C = inputShape[IO_CHANNEL_DIMENSION] + pads_begin[1] + pads_end[1];
                    size_t H = inputShape[IO_HEIGHT_DIMENSION]  + pads_begin[2] + pads_end[2];
                    size_t W = inputShape[IO_WIDTH_DIMENSION]   + pads_begin[3] + pads_end[3];

                    Shape outputShape({W, H, C, N});

                    outputs.emplace_back(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder());
                };
    }

namespace op {
        MV_REGISTER_OP(Pad)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<std::array<unsigned short, 4>>("pads_begin")
        .setArg<std::array<unsigned short, 4>>("pads_end")
        .setArg<std::string>("pad_mode")
        .setOptionalArg<double>("pad_value", 0.0)
        .setInputCheck(op_pad::inputCheckFcn)
        .setOutputDef(op_pad::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
}

}
