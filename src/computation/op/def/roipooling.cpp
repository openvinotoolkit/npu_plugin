#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_roi_pooling
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto input = inputs[0];
            if (inputs.size() != 2)
            {
                std::stringstream err;
                err << "Incorrect number of inputs (must be 2): " << inputs.size();
                errMsg = err.str();
                return {false, 0};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&, std::vector<Tensor>& outputs)
        {

            auto input = inputs[0];
            auto outputOrder = input->getOrder();
            auto inputShape = input->getShape();
            auto ndims = inputShape.ndims();
            mv::Shape outputShape(ndims);

            outputs.push_back(mv::Tensor(":0", input->getShape(), input->getDType(), input->getOrder()));

        };
    }
    namespace op
    {
        MV_REGISTER_OP(ROIPooling)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setArg<unsigned>("pooled_w")
        .setArg<unsigned>("pooled_h")
        .setArg<double>("spatial_scale")
        .setArg<unsigned>("roi_pooling_method")
        .setArg<unsigned>("num_rois")
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_roi_pooling::inputCheckFcn)
        .setOutputDef(op_roi_pooling::outputDefFcn)
        .setTypeTrait({"executable", "exposed"})
        .setVariableInputNum(true);
    }

}
