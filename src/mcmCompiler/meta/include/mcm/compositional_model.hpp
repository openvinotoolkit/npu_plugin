/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#ifndef MV_COMPOSITIONAL_MODEL_HPP_
#define MV_COMPOSITIONAL_MODEL_HPP_

#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"

#include "include/mcm/tensor/quantization_params.hpp"

namespace mv

{

    class CompositionalModel
    {

    public:

        virtual ~CompositionalModel() = 0;

        virtual mv::Data::TensorIterator align(Data::TensorIterator data, const std::size_t& dimension = 2, const std::size_t& pad = 16, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator argmax(Data::TensorIterator data, const int64_t& out_max_val, const int64_t& top_k, const int64_t& axis = 99, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator averagePool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator batchNormalization(Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator bias(Data::TensorIterator data, Data::TensorIterator weights, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator cTCDecoder(Data::TensorIterator data, Data::TensorIterator seq, const bool& ctc_merge_repeated, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator concat(const std::vector< Data::TensorIterator >& inputs, const std::string& axis = "C", const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator constant(const std::vector<double>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator constantInt(const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator conv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator copy(Data::TensorIterator data, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator crop(Data::TensorIterator data, const std::size_t& cropVal, const std::size_t& dimension = 2, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator custom(const std::vector< Data::TensorIterator >& inputs, const std::vector<uint8_t>& kernelData, const std::vector<uint8_t>& paramData, const Order& outOrder, const Shape& outShape, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator deconv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1, const bool& is_depthwise = false, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator depthwiseConv(Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator detectionOutput(const std::vector< Data::TensorIterator >& inputs, const int64_t& num_classes, const int64_t& keep_top_k, const double& nms_threshold, const int64_t& background_label_id, const int64_t& top_k, const bool& variance_encoded_in_target, const std::string& code_type, const bool& share_location, const double& confidence_threshold, const bool& clip_before_nms, const bool& clip_after_nms, const int64_t& decrease_label_id, const bool& normalized, const int64_t& input_height, const int64_t& input_width, const double& objectness_score, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator dropout(Data::TensorIterator input, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator dummy(Data::TensorIterator data, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator eltwise(const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator elu(Data::TensorIterator data, const unsigned& alpha = 1, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator fakeQuantize(Data::TensorIterator data, Data::TensorIterator input_min, Data::TensorIterator input_max, Data::TensorIterator output_min, Data::TensorIterator output_max, const unsigned& levels, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator flatten(Data::TensorIterator input, const int64_t& axis = 1, const int64_t& end_axis = 3, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator fullyConnected(Data::TensorIterator data, Data::TensorIterator weights, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator identity(Data::TensorIterator data, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator implicitInput(Data::TensorIterator data, const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator implicitOutput(Data::TensorIterator data, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator input(const Shape& shape, const DType& dType, const Order& order, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const bool& networkInput = true, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator interp(Data::TensorIterator data, const double& factor, const unsigned& pad_beg, const unsigned& pad_end, const unsigned& height = 0, const unsigned& width = 0, const bool& align_corners = true, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator leakyRelu(Data::TensorIterator data, const double& alpha = 0, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator localResponseNormalization(Data::TensorIterator data, const unsigned& size, const unsigned& bias, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator matMul(Data::TensorIterator data0, Data::TensorIterator data1, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator maxPool(Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator maximum(Data::TensorIterator inputs, const double& maximum, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator minimum(Data::TensorIterator inputs, const double& minimum, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator norm(Data::TensorIterator data, const double& alpha, const double& beta, const std::string& region, const unsigned& local_size, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator normalize(Data::TensorIterator data, Data::TensorIterator weights, const double& eps, const unsigned& across_spatial = 0, const unsigned& channel_shared = 0, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator output(Data::TensorIterator data, const DType& precision = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const bool& networkOutput = true, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator pSROIPooling(const std::vector< Data::TensorIterator >& inputs, const std::size_t& output_dim, const std::size_t& group_size, const double& spatial_scale, const std::size_t& pooled_w, const std::size_t& pooled_h, const std::size_t& spatial_bin_x, const std::size_t& spatial_bin_y, const std::string& mode, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator permute(Data::TensorIterator data, const Order& order, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator prelu(Data::TensorIterator data, Data::TensorIterator slope, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator priorbox(const std::vector< Data::TensorIterator >& inputs, const unsigned& flip, const unsigned& clip, const double& step_w, const double& step_h, const double& offset, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator proposal(const std::vector< Data::TensorIterator >& inputs, const std::vector<float>& scale, const std::vector<float>& ratio, const unsigned& base_size, const unsigned& pre_nms_topn, const unsigned& post_nms_topn, const double& nms_thresh, const unsigned& feat_stride, const unsigned& min_size, const double& pre_nms_thresh = 0.000000000000000, const bool& clip_before_nms = true, const bool& clip_after_nms = false, const bool& normalize = false, const double& box_size_scale = 1.000000000000000, const double& box_coordinate_scale = 1.000000000000000, const std::string& framework = "TENSORFLOW", const bool& for_deformable = false, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator quantize(Data::TensorIterator data, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator rOIPooling(const std::vector< Data::TensorIterator >& inputs, const unsigned& pooled_w, const unsigned& pooled_h, const double& spatial_scale, const unsigned& roi_pooling_method, const unsigned& num_rois, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator regionYolo(Data::TensorIterator data, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num = 0, const std::vector<unsigned>& mask = {}, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator relu(Data::TensorIterator data, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator reorder(Data::TensorIterator data, const Order& order, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator reorgYolo(Data::TensorIterator data, const unsigned& stride, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator resample(Data::TensorIterator input, const std::string& interpolation, const bool& antialias, const Shape& output_shape, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator reshape(Data::TensorIterator data, const Shape& shape, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator scale(Data::TensorIterator data, Data::TensorIterator weights, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator sigmoid(Data::TensorIterator data, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator slice(Data::TensorIterator data, const Shape& begin, const Shape& size, const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator softmax(Data::TensorIterator data, const std::string& axis = "C", const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator tanh(Data::TensorIterator data, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator tile(Data::TensorIterator data, const unsigned& axis, const unsigned& tiles, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;
        virtual mv::Data::TensorIterator topK(Data::TensorIterator data, const std::string& sort, const std::string& mode, const int64_t& top_k, const int64_t& axis = 99, const DType& dType = mv::DType("Default"), const mv::QuantizationParams& quantParams = {{},{},{},{}}, const std::string& name = "") = 0;

        virtual Data::OpListIterator getSourceOp(Data::TensorIterator tensor) = 0;
        virtual void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) = 0;
        virtual bool isValid() const = 0;
        virtual bool isValid(Data::TensorIterator tensor) const = 0;
        virtual bool isValid(Data::OpListIterator op) const = 0;
        virtual std::string getName() const = 0;

    };

}

#endif //MV_COMPOSITIONAL_MODEL_HPP_
