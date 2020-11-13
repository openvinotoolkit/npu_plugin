#include "include/mcm/computation/op/op_registry.hpp"

namespace mv {

namespace op_strided_slice {

static std::function<std::pair<bool, std::size_t>(
    const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, std::string&)>
    inputCheckFcn = [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
                        std::string& errMsg) -> std::pair<bool, std::size_t> {
    auto begin = args.at("begin").get<mv::Shape>();
    auto stride = args.at("stride").get<mv::Shape>();
    auto end = args.at("end").get<mv::Shape>();

    if (begin.ndims() != 4 || stride.ndims() != 4 || end.ndims() != 4) {
        std::stringstream ss;
        ss << "Begin, stride, and end shapes need to have dimensions of 4. Begin - " << begin.ndims() << ", End - "
           << end.ndims() << ", Stride - " << stride.ndims() << ".";
        errMsg = ss.str();
        return {false, 0};
    }

    for (size_t i = 0; i < begin.ndims(); ++i) {
        if (begin[i] >= end[i]) {
            errMsg = "Begin shape must be smaller than end shape element wise.";
            return {false, 0};
        }
    }

    return {true, 0};
};

static std::function<void(
    const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, std::vector<Tensor>&)>
    outputDefFcn = [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
                       std::vector<Tensor>& outputs) {
        auto begin = args.at("begin").get<mv::Shape>();
        auto stride = args.at("stride").get<mv::Shape>();
        auto end = args.at("end").get<mv::Shape>();

        std::vector<size_t> outputShapeVector;
        for (size_t i = 0; i < begin.ndims(); ++i) {
            outputShapeVector.push_back((end[i] - begin[i]) / stride[i]);
        }
        mv::Shape outputShapeSize(outputShapeVector);

        outputs.emplace_back(":0", outputShapeSize, inputs[0]->getDType(), inputs[0]->getOrder());

        if (inputs[0]->isPopulated()) {
            const size_t KW = mv::KERNEL_WIDTH;
            const size_t KH = mv::KERNEL_HEIGHT;
            const size_t IC = mv::KERNEL_INPUT_CHANNELS;
            const size_t OC = mv::KERNEL_OUTPUT_CHANNELS;

            for (size_t oc = 0; oc < outputShapeSize[OC]; ++oc) {
                for (size_t ic = 0; ic < outputShapeSize[IC]; ++ic) {
                    for (size_t kh = 0; kh < outputShapeSize[KH]; ++kh) {
                        for (size_t kw = 0; kw < outputShapeSize[KW]; ++kw) {
                            outputs[0].at({kw, kh, ic, oc}) = inputs[0]->at({kw * stride[KW] + begin[KW],
                                kh * stride[KH] + begin[KH], ic * stride[IC] + begin[IC], oc * stride[OC] + begin[OC]});
                        }
                    }
                }
            }
        }
    };
}  // namespace op_strided_slice

namespace op {

MV_REGISTER_OP(StridedSlice)
    .setInputs({"data"})
    .setOutputs({"output"})
    .setArg<mv::Shape>("begin")
    .setArg<mv::Shape>("end")
    .setArg<mv::Shape>("stride")
    .setInputCheck(op_strided_slice::inputCheckFcn)
    .setOutputDef(op_strided_slice::outputDefFcn)
    .setTypeTrait({"exposed"});
}

}  // namespace mv
