/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#include "/home/johnbrady/git/mcmCompiler/build/meta/include/mcm/op_model.hpp"

mv::OpModel::OpModel(const std::string& name) :
BaseOpModel(name)
{

}

mv::OpModel::OpModel(ComputationModel& other) :
BaseOpModel(other)
{

}

mv::OpModel::~OpModel()
{

}

mv::Data::TensorIterator mv::OpModel::align(Data::TensorIterator data, const std::size_t& dimension, const std::size_t& pad, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Align",
        {
            data
        },
        {
            { "dimension", dimension },
            { "pad", pad },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::argmax(Data::TensorIterator data, const int64_t& out_max_val, const int64_t& top_k, const int64_t& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Argmax",
        {
            data
        },
        {
            { "out_max_val", out_max_val },
            { "top_k", top_k },
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const std::string& auto_pad, const std::string& rounding_type, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "AveragePool",
        {
            data
        },
        {
            { "kSize", kSize },
            { "stride", stride },
            { "padding", padding },
            { "exclude_pad", exclude_pad },
            { "auto_pad", auto_pad },
            { "rounding_type", rounding_type },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::barrierTask(const Barrier& Barrier, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "BarrierTask",
        {
        },
        {
            { "Barrier", Barrier }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "BatchNormalization",
        {
            data,
            mean,
            variance,
            offset,
            scale
        },
        {
            { "eps", eps },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::bias(Data::TensorIterator data, Data::TensorIterator weights, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Bias",
        {
            data,
            weights
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::concat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Concat",
        inputs,
        {
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::constant(const std::vector<double>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Constant",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::constantDataElement(const std::vector<mv::DataElement>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "ConstantDataElement",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::constantInt(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "ConstantInt",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Conv",
        {
            data,
            weights
        },
        {
            { "stride", stride },
            { "padding", padding },
            { "dilationFactor", dilationFactor },
            { "group", group },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::conversion(Data::TensorIterator data, const Order& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Conversion",
        {
            data
        },
        {
            { "order", order }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::copy(Data::TensorIterator data, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Copy",
        {
            data
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::crop(Data::TensorIterator data, const std::size_t& cropVal, const std::size_t& dimension, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Crop",
        {
            data
        },
        {
            { "cropVal", cropVal },
            { "dimension", dimension },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::dMATask(Data::TensorIterator data, const DmaDirection& direction, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DMATask",
        {
            data
        },
        {
            { "direction", direction }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::dPUTaskConv(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const unsigned& group, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("Conv") },
            { "stride", stride },
            { "padding", padding },
            { "dilationFactor", dilationFactor },
            { "group", group },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::dPUTaskMaxPool(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const std::string& auto_pad, const std::string& rounding_type, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("MaxPool") },
            { "kSize", kSize },
            { "stride", stride },
            { "padding", padding },
            { "exclude_pad", exclude_pad },
            { "auto_pad", auto_pad },
            { "rounding_type", rounding_type },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::dPUTaskDepthwiseConv(const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("DepthwiseConv") },
            { "stride", stride },
            { "padding", padding },
            { "dilationFactor", dilationFactor },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::dPUTaskEltwise(const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DPUTask",
        inputs,
        {
            { "taskOp", std::string("Eltwise") },
            { "eltwiseType", eltwiseType },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::deallocate(Data::TensorIterator inputs, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Deallocate",
        {
            inputs
        },
        {
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::depthwiseConv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DepthwiseConv",
        {
            data,
            weights
        },
        {
            { "stride", stride },
            { "padding", padding },
            { "dilationFactor", dilationFactor },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::detectionOutput(const std::vector< Data::TensorIterator >& inputs, const int64_t& num_classes, const int64_t& keep_top_k, const double& nms_threshold, const int64_t& background_label_id, const int64_t& top_k, const bool& variance_encoded_in_target, const std::string& code_type, const bool& share_location, const double& confidence_threshold, const bool& clip_before_nms, const bool& clip_after_nms, const int64_t& decrease_label_id, const bool& normalized, const int64_t& input_height, const int64_t& input_width, const double& objectness_score, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "DetectionOutput",
        inputs,
        {
            { "num_classes", num_classes },
            { "keep_top_k", keep_top_k },
            { "nms_threshold", nms_threshold },
            { "background_label_id", background_label_id },
            { "top_k", top_k },
            { "variance_encoded_in_target", variance_encoded_in_target },
            { "code_type", code_type },
            { "share_location", share_location },
            { "confidence_threshold", confidence_threshold },
            { "clip_before_nms", clip_before_nms },
            { "clip_after_nms", clip_after_nms },
            { "decrease_label_id", decrease_label_id },
            { "normalized", normalized },
            { "input_height", input_height },
            { "input_width", input_width },
            { "objectness_score", objectness_score },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::dropout(Data::TensorIterator input, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Dropout",
        {
            input
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::dummy(Data::TensorIterator data, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Dummy",
        {
            data
        },
        {
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::eltwise(const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Eltwise",
        inputs,
        {
            { "eltwiseType", eltwiseType },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::elu(Data::TensorIterator data, const unsigned& alpha, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Elu",
        {
            data
        },
        {
            { "alpha", alpha }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::flatten(Data::TensorIterator input, const int64_t& axis, const int64_t& end_axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Flatten",
        {
            input
        },
        {
            { "axis", axis },
            { "end_axis", end_axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::fullyConnected(Data::TensorIterator data, Data::TensorIterator weights, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "FullyConnected",
        {
            data,
            weights
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::identity(Data::TensorIterator data, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Identity",
        {
            data
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::implicitConcat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "ImplicitConcat",
        inputs,
        {
            { "axis", axis },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::implicitReshape(Data::TensorIterator inputs, const Shape& shape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "ImplicitReshape",
        {
            inputs
        },
        {
            { "shape", shape },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::input(const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Input",
        {
        },
        {
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::interp(Data::TensorIterator data, const double& factor, const unsigned& pad_beg, const unsigned& pad_end, const unsigned& height, const unsigned& width, const bool& align_corners, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Interp",
        {
            data
        },
        {
            { "factor", factor },
            { "pad_beg", pad_beg },
            { "pad_end", pad_end },
            { "height", height },
            { "width", width },
            { "align_corners", align_corners },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::leakyRelu(Data::TensorIterator data, const double& alpha, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "LeakyRelu",
        {
            data
        },
        {
            { "alpha", alpha },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::localResponseNormalization(Data::TensorIterator data, const unsigned& size, const unsigned& bias, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "LocalResponseNormalization",
        {
            data
        },
        {
            { "size", size },
            { "bias", bias }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::matMul(Data::TensorIterator data0, Data::TensorIterator data1, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "MatMul",
        {
            data0,
            data1
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::maxPool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad, const std::string& auto_pad, const std::string& rounding_type, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "MaxPool",
        {
            data
        },
        {
            { "kSize", kSize },
            { "stride", stride },
            { "padding", padding },
            { "exclude_pad", exclude_pad },
            { "auto_pad", auto_pad },
            { "rounding_type", rounding_type },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::maximum(Data::TensorIterator inputs, const double& maximum, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Maximum",
        {
            inputs
        },
        {
            { "maximum", maximum },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::minimum(Data::TensorIterator inputs, const double& minimum, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Minimum",
        {
            inputs
        },
        {
            { "minimum", minimum },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::norm(Data::TensorIterator data, const double& alpha, const double& beta, const std::string& region, const unsigned& local_size, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Norm",
        {
            data
        },
        {
            { "alpha", alpha },
            { "beta", beta },
            { "region", region },
            { "local_size", local_size },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::normalize(Data::TensorIterator data, Data::TensorIterator weights, const double& eps, const unsigned& across_spatial, const unsigned& channel_shared, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Normalize",
        {
            data,
            weights
        },
        {
            { "eps", eps },
            { "across_spatial", across_spatial },
            { "channel_shared", channel_shared },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::output(Data::TensorIterator data, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Output",
        {
            data
        },
        {
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::permute(Data::TensorIterator data, const Order& order, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Permute",
        {
            data
        },
        {
            { "order", order },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::placeholderTask(const Shape& shape, const DType& dType, const Order& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "PlaceholderTask",
        {
        },
        {
            { "shape", shape },
            { "dType", dType },
            { "order", order }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::prelu(Data::TensorIterator data, Data::TensorIterator slope, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Prelu",
        {
            data,
            slope
        },
        {
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::priorbox(const std::vector< Data::TensorIterator >& inputs, const unsigned& flip, const unsigned& clip, const double& step_w, const double& step_h, const double& offset, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Priorbox",
        inputs,
        {
            { "flip", flip },
            { "clip", clip },
            { "step_w", step_w },
            { "step_h", step_h },
            { "offset", offset },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::proposal(const std::vector< Data::TensorIterator >& inputs, const unsigned& base_size, const unsigned& pre_nms_topn, const unsigned& post_nms_topn, const double& nms_thresh, const unsigned& feat_stride, const unsigned& min_size, const double& pre_nms_thresh, const bool& clip_before_nms, const bool& clip_after_nms, const bool& normalize, const double& box_size_scale, const double& box_coordinate_scale, const std::string& framework, const bool& for_deformable, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Proposal",
        inputs,
        {
            { "base_size", base_size },
            { "pre_nms_topn", pre_nms_topn },
            { "post_nms_topn", post_nms_topn },
            { "nms_thresh", nms_thresh },
            { "feat_stride", feat_stride },
            { "min_size", min_size },
            { "pre_nms_thresh", pre_nms_thresh },
            { "clip_before_nms", clip_before_nms },
            { "clip_after_nms", clip_after_nms },
            { "normalize", normalize },
            { "box_size_scale", box_size_scale },
            { "box_coordinate_scale", box_coordinate_scale },
            { "framework", framework },
            { "for_deformable", for_deformable },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::quantize(Data::TensorIterator data, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Quantize",
        {
            data
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::rOIPooling(const std::vector< Data::TensorIterator >& inputs, const unsigned& pooled_w, const unsigned& pooled_h, const double& spatial_scale, const unsigned& roi_pooling_method, const unsigned& num_rois, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "ROIPooling",
        inputs,
        {
            { "pooled_w", pooled_w },
            { "pooled_h", pooled_h },
            { "spatial_scale", spatial_scale },
            { "roi_pooling_method", roi_pooling_method },
            { "num_rois", num_rois },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false
    
    );
}

mv::Data::TensorIterator mv::OpModel::regionYolo(Data::TensorIterator data, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num, const std::vector<unsigned>& mask, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "RegionYolo",
        {
            data
        },
        {
            { "coords", coords },
            { "classes", classes },
            { "do_softmax", do_softmax },
            { "num", num },
            { "mask", mask },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::relu(Data::TensorIterator data, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Relu",
        {
            data
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::reorder(Data::TensorIterator data, const Order& order, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Reorder",
        {
            data
        },
        {
            { "order", order }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::reorgYolo(Data::TensorIterator data, const unsigned& stride, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "ReorgYolo",
        {
            data
        },
        {
            { "stride", stride },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::reshape(Data::TensorIterator data, const Shape& shape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Reshape",
        {
            data
        },
        {
            { "shape", shape },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::scale(Data::TensorIterator data, Data::TensorIterator weights, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Scale",
        {
            data,
            weights
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::sigmoid(Data::TensorIterator data, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Sigmoid",
        {
            data
        },
        {
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::slice(Data::TensorIterator data, const Shape& begin, const Shape& size, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Slice",
        {
            data
        },
        {
            { "begin", begin },
            { "size", size },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::softmax(Data::TensorIterator data, const std::string& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Softmax",
        {
            data
        },
        {
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::sparsityMap(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "SparsityMap",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::tanh(Data::TensorIterator data, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "Tanh",
        {
            data
        },
        {
        },
        name
    
    
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskDummy(const std::vector< Data::TensorIterator >& inputs, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Dummy") },
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskIdentity(const std::vector< Data::TensorIterator >& inputs, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Identity") },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskSoftmax(const std::vector< Data::TensorIterator >& inputs, const std::string& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Softmax") },
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskProposal(const std::vector< Data::TensorIterator >& inputs, const unsigned& base_size, const unsigned& pre_nms_topn, const unsigned& post_nms_topn, const double& nms_thresh, const unsigned& feat_stride, const unsigned& min_size, const double& pre_nms_thresh, const bool& clip_before_nms, const bool& clip_after_nms, const bool& normalize, const double& box_size_scale, const double& box_coordinate_scale, const std::string& framework, const bool& for_deformable, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Proposal") },
            { "base_size", base_size },
            { "pre_nms_topn", pre_nms_topn },
            { "post_nms_topn", post_nms_topn },
            { "nms_thresh", nms_thresh },
            { "feat_stride", feat_stride },
            { "min_size", min_size },
            { "pre_nms_thresh", pre_nms_thresh },
            { "clip_before_nms", clip_before_nms },
            { "clip_after_nms", clip_after_nms },
            { "normalize", normalize },
            { "box_size_scale", box_size_scale },
            { "box_coordinate_scale", box_coordinate_scale },
            { "framework", framework },
            { "for_deformable", for_deformable },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskROIPooling(const std::vector< Data::TensorIterator >& inputs, const unsigned& pooled_w, const unsigned& pooled_h, const double& spatial_scale, const unsigned& roi_pooling_method, const unsigned& num_rois, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("ROIPooling") },
            { "pooled_w", pooled_w },
            { "pooled_h", pooled_h },
            { "spatial_scale", spatial_scale },
            { "roi_pooling_method", roi_pooling_method },
            { "num_rois", num_rois },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskQuantize(const std::vector< Data::TensorIterator >& inputs, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Quantize") },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskReshape(const std::vector< Data::TensorIterator >& inputs, const Shape& shape, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Reshape") },
            { "shape", shape },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskRegionYolo(const std::vector< Data::TensorIterator >& inputs, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num, const std::vector<unsigned>& mask, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("RegionYolo") },
            { "coords", coords },
            { "classes", classes },
            { "do_softmax", do_softmax },
            { "num", num },
            { "mask", mask },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskReorgYolo(const std::vector< Data::TensorIterator >& inputs, const unsigned& stride, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("ReorgYolo") },
            { "stride", stride },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskNormalize(const std::vector< Data::TensorIterator >& inputs, const double& eps, const unsigned& across_spatial, const unsigned& channel_shared, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Normalize") },
            { "eps", eps },
            { "across_spatial", across_spatial },
            { "channel_shared", channel_shared },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskPermute(const std::vector< Data::TensorIterator >& inputs, const Order& order, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Permute") },
            { "order", order },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskEltwise(const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Eltwise") },
            { "eltwiseType", eltwiseType },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskInterp(const std::vector< Data::TensorIterator >& inputs, const double& factor, const unsigned& pad_beg, const unsigned& pad_end, const unsigned& height, const unsigned& width, const bool& align_corners, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Interp") },
            { "factor", factor },
            { "pad_beg", pad_beg },
            { "pad_end", pad_end },
            { "height", height },
            { "width", width },
            { "align_corners", align_corners },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskDetectionOutput(const std::vector< Data::TensorIterator >& inputs, const int64_t& num_classes, const int64_t& keep_top_k, const double& nms_threshold, const int64_t& background_label_id, const int64_t& top_k, const bool& variance_encoded_in_target, const std::string& code_type, const bool& share_location, const double& confidence_threshold, const bool& clip_before_nms, const bool& clip_after_nms, const int64_t& decrease_label_id, const bool& normalized, const int64_t& input_height, const int64_t& input_width, const double& objectness_score, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("DetectionOutput") },
            { "num_classes", num_classes },
            { "keep_top_k", keep_top_k },
            { "nms_threshold", nms_threshold },
            { "background_label_id", background_label_id },
            { "top_k", top_k },
            { "variance_encoded_in_target", variance_encoded_in_target },
            { "code_type", code_type },
            { "share_location", share_location },
            { "confidence_threshold", confidence_threshold },
            { "clip_before_nms", clip_before_nms },
            { "clip_after_nms", clip_after_nms },
            { "decrease_label_id", decrease_label_id },
            { "normalized", normalized },
            { "input_height", input_height },
            { "input_width", input_width },
            { "objectness_score", objectness_score },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskPriorbox(const std::vector< Data::TensorIterator >& inputs, const unsigned& flip, const unsigned& clip, const double& step_w, const double& step_h, const double& offset, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Priorbox") },
            { "flip", flip },
            { "clip", clip },
            { "step_w", step_w },
            { "step_h", step_h },
            { "offset", offset },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskArgmax(const std::vector< Data::TensorIterator >& inputs, const int64_t& out_max_val, const int64_t& top_k, const int64_t& axis, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Argmax") },
            { "out_max_val", out_max_val },
            { "top_k", top_k },
            { "axis", axis },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::uPATaskNorm(const std::vector< Data::TensorIterator >& inputs, const double& alpha, const double& beta, const std::string& region, const unsigned& local_size, const DType& dType, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "UPATask",
        inputs,
        {
            { "taskOp", std::string("Norm") },
            { "alpha", alpha },
            { "beta", beta },
            { "region", region },
            { "local_size", local_size },
            { "dType", dType },
            { "quantParams", quantParams }
        },
        name,
        false,
        false
    );
}

mv::Data::TensorIterator mv::OpModel::weightsTable(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams, const std::string& name)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_COMP)
    return defineOp(
        "WeightsTable",
        {
        },
        {
            { "data", data },
            { "shape", shape },
            { "dType", dType },
            { "order", order },
            { "quantParams", quantParams }
        },
        name
    
    
    );
}

mv::Data::OpListIterator mv::OpModel::getSourceOp(Data::TensorIterator tensor)
{
    return BaseOpModel::getSourceOp(tensor);
}
void mv::OpModel::addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr)
{
    return BaseOpModel::addAttr(op, name, attr);
}
bool mv::OpModel::isValid() const
{
    return BaseOpModel::isValid();
}
bool mv::OpModel::isValid(Data::TensorIterator tensor) const
{
    return BaseOpModel::isValid(tensor);
}
bool mv::OpModel::isValid(Data::OpListIterator op) const
{
    return BaseOpModel::isValid(op);
}
std::string mv::OpModel::getName() const
{
    return BaseOpModel::getName();
}
