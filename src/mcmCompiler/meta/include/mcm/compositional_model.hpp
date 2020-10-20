/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#ifndef MV_COMPOSITIONAL_MODEL_HPP_
#define MV_COMPOSITIONAL_MODEL_HPP_

#include "include/mcm/computation/model/iterator/data_context.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"

#include "include/mcm/tensor/quantization_params.hpp"

#include "include/mcm/tensor/tensor_info.hpp"

namespace mv

{

    class CompositionalModel
    {

    public:

        virtual ~CompositionalModel() = 0;

        virtual mv::Data::TensorIterator align(const std::string& name, Data::TensorIterator data, const std::size_t& dimension = 2, const std::size_t& pad = 16) = 0;
        virtual mv::Data::TensorIterator argmax(const std::string& name, Data::TensorIterator data, const int64_t& out_max_val, const int64_t& top_k, const int64_t& axis = 99) = 0;
        virtual mv::Data::TensorIterator averagePool(const std::string& name, Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true) = 0;
        virtual mv::Data::TensorIterator batchNormalization(const std::string& name, Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps) = 0;
        virtual mv::Data::TensorIterator bias(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights) = 0;
        virtual mv::Data::TensorIterator cTCDecoder(const std::string& name, Data::TensorIterator data, Data::TensorIterator seq, const bool& ctc_merge_repeated) = 0;
        virtual mv::Data::TensorIterator concat(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& axis = "C") = 0;
        virtual mv::Data::TensorIterator constant(const std::string& name, const std::vector<double>& data, const Shape& shape, const DType& dType, const Order& order) = 0;
        virtual mv::Data::TensorIterator constantInt(const std::string& name, const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order) = 0;
        virtual mv::Data::TensorIterator conv(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1) = 0;
        virtual mv::Data::TensorIterator copy(const std::string& name, Data::TensorIterator data) = 0;
        virtual mv::Data::TensorIterator crop(const std::string& name, Data::TensorIterator data, const std::size_t& cropVal, const std::size_t& dimension = 2) = 0;
        virtual mv::Data::TensorIterator custom(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::vector<uint8_t>& kernelData, const std::vector<uint8_t>& paramData, const std::vector<mv::TensorInfo>& outputsInfo) = 0;
        virtual mv::Data::TensorIterator deconv(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1, const bool& is_depthwise = false) = 0;
        virtual mv::Data::TensorIterator depthwiseConv(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1) = 0;
        virtual mv::Data::TensorIterator detectionOutput(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const int64_t& num_classes, const int64_t& keep_top_k, const double& nms_threshold, const int64_t& background_label_id, const int64_t& top_k, const bool& variance_encoded_in_target, const std::string& code_type, const bool& share_location, const double& confidence_threshold, const bool& clip_before_nms, const bool& clip_after_nms, const int64_t& decrease_label_id, const bool& normalized, const int64_t& input_height, const int64_t& input_width, const double& objectness_score) = 0;
        virtual mv::Data::TensorIterator dropout(const std::string& name, Data::TensorIterator input) = 0;
        virtual mv::Data::TensorIterator dummy(const std::string& name, Data::TensorIterator data) = 0;
        virtual mv::Data::TensorIterator eltwise(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType) = 0;
        virtual mv::Data::TensorIterator elu(const std::string& name, Data::TensorIterator data, const unsigned& alpha = 1) = 0;
        virtual mv::Data::TensorIterator exp(const std::string& name, Data::TensorIterator data) = 0;
        virtual mv::Data::TensorIterator fakeQuantize(const std::string& name, Data::TensorIterator data, Data::TensorIterator input_min, Data::TensorIterator input_max, Data::TensorIterator output_min, Data::TensorIterator output_max, const unsigned& levels) = 0;
        virtual mv::Data::TensorIterator flatten(const std::string& name, Data::TensorIterator input, const int64_t& axis = 1, const int64_t& end_axis = 3) = 0;
        virtual mv::Data::TensorIterator fullyConnected(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights) = 0;
        virtual mv::Data::TensorIterator gather(const std::string& name, Data::TensorIterator data, Data::TensorIterator indices, const unsigned& axis) = 0;
        virtual mv::Data::TensorIterator identity(const std::string& name, Data::TensorIterator data) = 0;
        virtual mv::Data::TensorIterator implicitInput(const std::string& name, Data::TensorIterator data, const Shape& shape, const DType& dType, const Order& order) = 0;
        virtual mv::Data::TensorIterator implicitOutput(const std::string& name, Data::TensorIterator data) = 0;
        virtual mv::Data::TensorIterator input(const std::string& name, const Shape& shape, const DType& dType, const Order& order, const bool& networkInput = true) = 0;
        virtual mv::Data::TensorIterator interp(const std::string& name, Data::TensorIterator data, const double& factor, const unsigned& pad_beg, const unsigned& pad_end, const unsigned& height = 0, const unsigned& width = 0, const bool& align_corners = true) = 0;
        virtual mv::Data::TensorIterator leakyRelu(const std::string& name, Data::TensorIterator data, const double& alpha = 0) = 0;
        virtual mv::Data::TensorIterator localResponseNormalization(const std::string& name, Data::TensorIterator data, const unsigned& size, const unsigned& bias) = 0;
        virtual mv::Data::TensorIterator matMul(const std::string& name, Data::TensorIterator data0, Data::TensorIterator data1) = 0;
        virtual mv::Data::TensorIterator maxPool(const std::string& name, Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true) = 0;
        virtual mv::Data::TensorIterator maximum(const std::string& name, Data::TensorIterator inputs, const double& maximum) = 0;
        virtual mv::Data::TensorIterator minimum(const std::string& name, Data::TensorIterator inputs, const double& minimum) = 0;
        virtual mv::Data::TensorIterator norm(const std::string& name, Data::TensorIterator data, const double& alpha, const double& beta, const std::string& region, const unsigned& local_size) = 0;
        virtual mv::Data::TensorIterator normalize(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights, const double& eps, const unsigned& across_spatial = 0, const unsigned& channel_shared = 0) = 0;
        virtual mv::Data::TensorIterator output(const std::string& name, Data::TensorIterator data, const DType& precision = mv::DType("Default"), const bool& networkOutput = true) = 0;
        virtual mv::Data::TensorIterator pSROIPooling(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::size_t& output_dim, const std::size_t& group_size, const double& spatial_scale, const std::size_t& pooled_w, const std::size_t& pooled_h, const std::size_t& spatial_bin_x, const std::size_t& spatial_bin_y, const std::string& mode) = 0;
        virtual mv::Data::TensorIterator permute(const std::string& name, Data::TensorIterator data, const Order& order) = 0;
        virtual mv::Data::TensorIterator prelu(const std::string& name, Data::TensorIterator data, Data::TensorIterator slope) = 0;
        virtual mv::Data::TensorIterator priorbox(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& flip, const unsigned& clip, const double& step_w, const double& step_h, const double& offset) = 0;
        virtual mv::Data::TensorIterator proposal(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::vector<float>& scale, const std::vector<float>& ratio, const unsigned& base_size, const unsigned& pre_nms_topn, const unsigned& post_nms_topn, const double& nms_thresh, const unsigned& feat_stride, const unsigned& min_size, const double& pre_nms_thresh = 0.000000000000000, const bool& clip_before_nms = true, const bool& clip_after_nms = false, const bool& normalize = false, const double& box_size_scale = 1.000000000000000, const double& box_coordinate_scale = 1.000000000000000, const std::string& framework = "TENSORFLOW", const bool& for_deformable = false) = 0;
        virtual mv::Data::TensorIterator quantize(const std::string& name, Data::TensorIterator data) = 0;
        virtual mv::Data::TensorIterator rOIPooling(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& pooled_w, const unsigned& pooled_h, const double& spatial_scale, const unsigned& roi_pooling_method, const unsigned& num_rois) = 0;
        virtual mv::Data::TensorIterator reciprocal(const std::string& name, Data::TensorIterator data) = 0;
        virtual mv::Data::TensorIterator regionYolo(const std::string& name, Data::TensorIterator data, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num = 0, const std::vector<unsigned>& mask = {}) = 0;
        virtual mv::Data::TensorIterator relu(const std::string& name, Data::TensorIterator data) = 0;
        virtual mv::Data::TensorIterator reorder(const std::string& name, Data::TensorIterator data, const Order& order) = 0;
        virtual mv::Data::TensorIterator reorgYolo(const std::string& name, Data::TensorIterator data, const unsigned& stride) = 0;
        virtual mv::Data::TensorIterator resample(const std::string& name, Data::TensorIterator input, const std::string& interpolation, const bool& antialias, const Shape& output_shape) = 0;
        virtual mv::Data::TensorIterator reshape(const std::string& name, Data::TensorIterator data, const Shape& shape) = 0;
        virtual mv::Data::TensorIterator scale(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights) = 0;
        virtual mv::Data::TensorIterator sigmoid(const std::string& name, Data::TensorIterator data) = 0;
        virtual mv::Data::TensorIterator slice(const std::string& name, Data::TensorIterator data, const Shape& begin, const Shape& size) = 0;
        virtual mv::Data::TensorIterator softmax(const std::string& name, Data::TensorIterator data, const std::string& axis = "C") = 0;
        virtual mv::Data::TensorIterator tanh(const std::string& name, Data::TensorIterator data) = 0;
        virtual mv::Data::TensorIterator tile(const std::string& name, Data::TensorIterator data, const unsigned& axis, const unsigned& tiles) = 0;
        virtual mv::Data::TensorIterator topK(const std::string& name, Data::TensorIterator data, const std::string& sort, const std::string& mode, const int64_t& top_k, const int64_t& axis = 99) = 0;

        virtual Data::OpListIterator getSourceOp(Data::TensorIterator tensor) = 0;
        virtual void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) = 0;
        virtual bool isValid() const = 0;
        virtual bool isValid(Data::TensorIterator tensor) const = 0;
        virtual bool isValid(Data::OpListIterator op) const = 0;
        virtual std::string getName() const = 0;

    };

}

#endif //MV_COMPOSITIONAL_MODEL_HPP_
