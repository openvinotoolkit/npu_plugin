#include "include/mcm/computation/op/op_registry.hpp"

namespace mv {

namespace op_strided_slice {

static std::function<std::pair<bool, std::size_t>(
    const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, std::string&)>
    inputCheckFcn = [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
                        std::string& errMsg) -> std::pair<bool, std::size_t> {
    UNUSED(inputs);
    auto begin = args.at("begins").get<std::vector<unsigned>>();
    auto end = args.at("ends").get<std::vector<unsigned>>();
    auto strides = args.at("strides").get<std::vector<unsigned>>();

    for (size_t i = 0; i < begin.size(); ++i) {
        if (begin[i] >= end[i]) {
            errMsg = "Begin shape must be smaller than end shape element wise.";
            return {false, 0};
        }
    }

    for (size_t i = 0; i < strides.size(); ++i) {
        if (strides[i] <= 0) {
            errMsg = "Strides must be positive.";
            return {false, 0};
        }
    }

    return {true, 0};
};

static std::function<void(
    const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, std::vector<Tensor>&)>
    outputDefFcn = [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
                       std::vector<Tensor>& outputs) {
        auto out_shape = args.at("out_shape").get<std::vector<unsigned>>();

        auto outputShape = std::vector<size_t>(out_shape.size());
        for (size_t i = 0; i < out_shape.size(); ++i) {
            outputShape[i] = out_shape[i];
        }

        outputs.emplace_back(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder());
    };
}  // namespace op_strided_slice

namespace op {

MV_REGISTER_OP(StridedSlice)
    .setInputs({"data"})
    .setOutputs({"output"})
    .setArg<std::vector<unsigned>>("begins")
    .setArg<std::vector<unsigned>>("ends")
    .setArg<std::vector<unsigned>>("strides")
    .setArg<std::vector<unsigned>>("out_shape")
    .setInputCheck(op_strided_slice::inputCheckFcn)
    .setOutputDef(op_strided_slice::outputDefFcn)
    .setTypeTrait({"exposed"});
}

}  // namespace mv
