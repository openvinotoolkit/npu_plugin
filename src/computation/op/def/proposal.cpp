#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_proposal
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto input = inputs[0];
            if (inputs.size() != 3)
            {
                std::stringstream err;
                err << "Incorrect number of inputs (must be 3): " << inputs.size();
                errMsg = err.str();
                return {false, 0};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            auto input = inputs[0];
            auto outputOrder = input->getOrder();
            auto W = args.at("post_nms_topn").get<unsigned>();
            size_t H = 1;
            size_t C = 5;
            size_t N = 1;

            mv::Shape outputShape({W, H, C, N});;

            outputs.push_back(mv::Tensor(":0", outputShape, input->getDType(), input->getOrder()));

        };
    }

    namespace op
    {


        MV_REGISTER_OP(Proposal)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setArg<std::vector<float>>("scale")
        .setArg<std::vector<float>>("ratio")
        .setArg<unsigned>("base_size")
        .setArg<unsigned>("pre_nms_topn")
        .setArg<unsigned>("post_nms_topn")
        .setArg<double>("nms_thresh")
        .setArg<unsigned>("feat_stride")
        .setArg<unsigned>("min_size")
        .setOptionalArg<double>("pre_nms_thresh", 0.0)
        .setOptionalArg<bool>("clip_before_nms", true)
        .setOptionalArg<bool>("clip_after_nms", false)
        .setOptionalArg<bool>("normalize", false)
        .setOptionalArg<double>("box_size_scale", 1.0)
        .setOptionalArg<double>("box_coordinate_scale", 1.0)
        .setOptionalArg<std::string>("framework", std::string("TENSORFLOW"))
        .setOptionalArg<bool>("for_deformable", false)
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_proposal::inputCheckFcn)
        .setOutputDef(op_proposal::outputDefFcn)
        .setTypeTrait({"executable", "exposed"})
        .setVariableInputNum(true);
    }

}
