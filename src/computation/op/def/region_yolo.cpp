#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_region_yolo
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto input = inputs[0];

            if (inputs.size() != 1)
            {
                std::stringstream err;
                err << "Too many inputs (must be 1): " << inputs.size();
                errMsg = err.str();
                return {false, 0};
            }

            auto order = input->getOrder();
            auto order_str = order.toString();

            // Yolo does not support 3D images (like "NCDHW")
            if (order_str.find("D") != std::string::npos)
            {
                errMsg = "Depth dimension not supported, order=" + order_str;
                return {false, 0};
            }

            auto iC = input->getShape().getAxis("C");
            auto iH = input->getShape().getAxis("H");
            auto iW = input->getShape().getAxis("W");

            if (iC == std::string::npos)
            {
                errMsg = "Tensor must have channels dimension: order=" + order_str;
                return {false, 0};
            }

            auto shape = input->getShape();

            auto C = shape[iC];
            auto H = shape[iH];
            auto W = shape[iW];

            auto coords = args.at("coords").get<unsigned>();
            auto classes = args.at("classes").get<unsigned>();
            auto do_softmax = args.at("do_softmax").get<bool>();
            auto num = args.at("num").get<unsigned>();
            auto mask = args.at("mask").get<std::vector<unsigned>>();

            size_t _num_ = do_softmax ? num : mask.size();

            if (_num_ == 0)
            {
                if (do_softmax)
                    errMsg = "Incorrect optional parameter: num=0";
                else
                    errMsg = "Incorrect optional parameter: mask={}";

                return {false, 0};
            }

            auto output_size = H * W * _num_ * (classes + coords + 1);
            auto input_size  = H * W * C;

            if (input_size != output_size)
            {
                std::stringstream err;
                err << "Tensor size mismatch: order=" << order_str
                    << ", width=" << W << ", height=" << H << ", channels=" << C
                    << ", coords=" << coords << ", classes=" << classes
                    << ", do_softmax=" << do_softmax;

                if (do_softmax)
                {
                    err << ", num=" << num;
                }
                else
                {
                    err << ", mask size=" << mask.size();
                }

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

            auto in_order = input->getOrder();
            auto in_order_str = in_order.toString();

            auto out_order = in_order;
            auto out_shape = input->getShape();

            outputs.push_back(mv::Tensor(":0", out_shape, input->getDType(), out_order));
        };

        static std::vector<unsigned> empty;
    }

    namespace op {
        // Region Yolo converts tensor like e.g. N×C×H×W into Nx(C*H*W)
        // with following formula:
        // - number of output channels = H * W * _num_ * (classes + coords + 1)
        // - value of _num_ depends on parameters num and mask:
        //     _num_ = value of parameter num, if parameter do_softmax is true
        //     _num_ = size of mask vector otherwise
        // - num, mask, do_softmax, classes, coords are layer parameters
        //
        // Additional parameters axis and axis_end of IR layer are redundant
        // and so omitted, as one always knows the order of MCM tensor axes.


        MV_REGISTER_OP(RegionYolo)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<unsigned>("coords")
        .setArg<unsigned>("classes")
        .setArg<bool>("do_softmax")
        .setOptionalArg<unsigned>("num", 0)
        .setOptionalArg<std::vector<unsigned>>("mask", op_region_yolo::empty)
        .setOptionalArg<mv::DType>("dType", mv::DType("Default"))
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_region_yolo::inputCheckFcn)
        .setOutputDef(op_region_yolo::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}
