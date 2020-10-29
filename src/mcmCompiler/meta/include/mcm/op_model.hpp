/*
    DO NOT MODIFY - that file was generated automatically using op::OpRegistry::generateCompositionAPI()
*/

#ifndef MV_OP_MODEL_HPP_
#define MV_OP_MODEL_HPP_

#include "include/mcm/compositional_model.hpp"
#include "include/mcm/computation/model/base_op_model.hpp"

#include "include/mcm/compiler/compilation_profiler.hpp"

namespace mv

{

    class OpModel: public BaseOpModel, public CompositionalModel
    {

    private:
    public:

        OpModel(const std::string& name);
        OpModel(ComputationModel& model);
        virtual ~OpModel();

        mv::Data::TensorIterator align(const std::string& name, Data::TensorIterator data, const std::size_t& dimension = 2, const std::size_t& pad = 16) override;
        mv::Data::TensorIterator argmax(const std::string& name, Data::TensorIterator data, const int64_t& out_max_val, const int64_t& top_k, const int64_t& axis = 99) override;
        mv::Data::TensorIterator averagePool(const std::string& name, Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true) override;
        mv::Data::TensorIterator barrierTask(const std::string& name, const Barrier& Barrier);
        mv::Data::TensorIterator batchNormalization(const std::string& name, Data::TensorIterator data, Data::TensorIterator mean, Data::TensorIterator variance, Data::TensorIterator offset, Data::TensorIterator scale, const double& eps) override;
        mv::Data::TensorIterator bias(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights) override;
        mv::Data::TensorIterator cTCDecoder(const std::string& name, Data::TensorIterator data, Data::TensorIterator seq, const bool& ctc_merge_repeated) override;
        mv::Data::TensorIterator concat(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& axis = "C") override;
        mv::Data::TensorIterator constant(const std::string& name, const std::vector<double>& data, const Shape& shape, const DType& dType, const Order& order) override;
        mv::Data::TensorIterator constantDataElement(const std::string& name, const std::vector<mv::DataElement>& data, const Shape& shape, const DType& dType, const Order& order);
        mv::Data::TensorIterator constantInt(const std::string& name, const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order) override;
        mv::Data::TensorIterator conv(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1) override;
        mv::Data::TensorIterator conversion(const std::string& name, Data::TensorIterator data, const Order& order);
        mv::Data::TensorIterator copy(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator crop(const std::string& name, Data::TensorIterator data, const std::size_t& cropVal, const std::size_t& dimension = 2) override;
        mv::Data::TensorIterator custom(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::vector<uint8_t>& kernelData, const std::vector<uint8_t>& paramData, const std::vector<mv::TensorInfo>& outputsInfo) override;
        mv::Data::TensorIterator dMATask(const std::string& name, Data::TensorIterator data, const DmaDirection& direction, const uint8_t& port = 0);
        mv::Data::TensorIterator dPUTaskConv(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1);
mv::Data::TensorIterator dPUTaskMaxPool(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true);
mv::Data::TensorIterator dPUTaskDepthwiseConv(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1);
mv::Data::TensorIterator dPUTaskEltwise(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType);
        mv::Data::TensorIterator deallocate(const std::string& name, Data::TensorIterator inputs);
        mv::Data::TensorIterator deconv(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1, const bool& is_depthwise = false) override;
        mv::Data::TensorIterator depthwiseConv(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1) override;
        mv::Data::TensorIterator detectionOutput(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const int64_t& num_classes, const int64_t& keep_top_k, const double& nms_threshold, const int64_t& background_label_id, const int64_t& top_k, const bool& variance_encoded_in_target, const std::string& code_type, const bool& share_location, const double& confidence_threshold, const bool& clip_before_nms, const bool& clip_after_nms, const int64_t& decrease_label_id, const bool& normalized, const int64_t& input_height, const int64_t& input_width, const double& objectness_score) override;
        mv::Data::TensorIterator dropout(const std::string& name, Data::TensorIterator input) override;
        mv::Data::TensorIterator dummy(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator eltwise(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType) override;
        mv::Data::TensorIterator elu(const std::string& name, Data::TensorIterator data, const unsigned& alpha = 1) override;
        mv::Data::TensorIterator exp(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator fakeQuantize(const std::string& name, Data::TensorIterator data, Data::TensorIterator input_min, Data::TensorIterator input_max, Data::TensorIterator output_min, Data::TensorIterator output_max, const unsigned& levels) override;
        mv::Data::TensorIterator flatten(const std::string& name, Data::TensorIterator input, const int64_t& axis = 1, const int64_t& end_axis = 3) override;
        mv::Data::TensorIterator fullyConnected(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights) override;
        mv::Data::TensorIterator gather(const std::string& name, Data::TensorIterator data, Data::TensorIterator indices, const unsigned& axis) override;
        mv::Data::TensorIterator hSwish(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator identity(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator implicitConcat(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& axis = "C");
        mv::Data::TensorIterator implicitInput(const std::string& name, Data::TensorIterator data, const Shape& shape, const DType& dType, const Order& order) override;
        mv::Data::TensorIterator implicitInputSlice(const std::string& name, Data::TensorIterator inputs);
        mv::Data::TensorIterator implicitJoin(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& axis = "HW");
        mv::Data::TensorIterator implicitOutput(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator implicitPermute(const std::string& name, Data::TensorIterator inputs, const Shape& shape);
        mv::Data::TensorIterator implicitReshape(const std::string& name, Data::TensorIterator inputs, const Shape& shape);
        mv::Data::TensorIterator implicitUnion(const std::string& name, const std::vector< Data::TensorIterator >& inputs);
        mv::Data::TensorIterator input(const std::string& name, const Shape& shape, const DType& dType, const Order& order, const bool& networkInput = true) override;
        mv::Data::TensorIterator interp(const std::string& name, Data::TensorIterator data, const double& factor, const unsigned& pad_beg, const unsigned& pad_end, const unsigned& height = 0, const unsigned& width = 0, const bool& align_corners = true) override;
        mv::Data::TensorIterator leakyRelu(const std::string& name, Data::TensorIterator data, const double& alpha = 0) override;
        mv::Data::TensorIterator localResponseNormalization(const std::string& name, Data::TensorIterator data, const unsigned& size, const unsigned& bias) override;
        mv::Data::TensorIterator matMul(const std::string& name, Data::TensorIterator data0, Data::TensorIterator data1) override;
        mv::Data::TensorIterator maxPool(const std::string& name, Data::TensorIterator data, const std::array<unsigned short, 2>& kSize, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const bool& exclude_pad = true) override;
        mv::Data::TensorIterator maximum(const std::string& name, Data::TensorIterator inputs, const double& maximum) override;
        mv::Data::TensorIterator minimum(const std::string& name, Data::TensorIterator inputs, const double& minimum) override;
        mv::Data::TensorIterator norm(const std::string& name, Data::TensorIterator data, const double& alpha, const double& beta, const std::string& region, const unsigned& local_size) override;
        mv::Data::TensorIterator normalize(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights, const double& eps, const unsigned& across_spatial = 0, const unsigned& channel_shared = 0) override;
        mv::Data::TensorIterator output(const std::string& name, Data::TensorIterator data, const DType& precision = mv::DType("Default"), const bool& networkOutput = true) override;
        mv::Data::TensorIterator pSROIPooling(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::size_t& output_dim, const std::size_t& group_size, const double& spatial_scale, const std::size_t& pooled_w, const std::size_t& pooled_h, const std::size_t& spatial_bin_x, const std::size_t& spatial_bin_y, const std::string& mode) override;
        mv::Data::TensorIterator permute(const std::string& name, Data::TensorIterator data, const Order& order) override;
        mv::Data::TensorIterator placeholderTask(const std::string& name, const Shape& shape, const DType& dType, const Order& order);
        mv::Data::TensorIterator prelu(const std::string& name, Data::TensorIterator data, Data::TensorIterator slope) override;
        mv::Data::TensorIterator priorbox(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& flip, const unsigned& clip, const double& step_w, const double& step_h, const double& offset) override;
        mv::Data::TensorIterator proposal(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::vector<float>& scale, const std::vector<float>& ratio, const unsigned& base_size, const unsigned& pre_nms_topn, const unsigned& post_nms_topn, const double& nms_thresh, const unsigned& feat_stride, const unsigned& min_size, const double& pre_nms_thresh = 0.000000000000000, const bool& clip_before_nms = true, const bool& clip_after_nms = false, const bool& normalize = false, const double& box_size_scale = 1.000000000000000, const double& box_coordinate_scale = 1.000000000000000, const std::string& framework = "TENSORFLOW", const bool& for_deformable = false) override;
        mv::Data::TensorIterator pseudoOp(const std::string& name, const std::vector< Data::TensorIterator >& inputs);
        mv::Data::TensorIterator quantize(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator rOIPooling(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& pooled_w, const unsigned& pooled_h, const double& spatial_scale, const unsigned& roi_pooling_method, const unsigned& num_rois) override;
        mv::Data::TensorIterator reciprocal(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator refConv(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1);
        mv::Data::TensorIterator regionYolo(const std::string& name, Data::TensorIterator data, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num = 0, const std::vector<unsigned>& mask = {}) override;
        mv::Data::TensorIterator relu(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator reorder(const std::string& name, Data::TensorIterator data, const Order& order) override;
        mv::Data::TensorIterator reorgYolo(const std::string& name, Data::TensorIterator data, const unsigned& stride) override;
        mv::Data::TensorIterator resample(const std::string& name, Data::TensorIterator input, const std::string& interpolation, const bool& antialias, const Shape& output_shape) override;
        mv::Data::TensorIterator reshape(const std::string& name, Data::TensorIterator data, const Shape& shape) override;
        mv::Data::TensorIterator scale(const std::string& name, Data::TensorIterator data, Data::TensorIterator weights) override;
        mv::Data::TensorIterator sigmoid(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator slice(const std::string& name, Data::TensorIterator data, const Shape& begin, const Shape& size) override;
        mv::Data::TensorIterator softmax(const std::string& name, Data::TensorIterator data, const std::string& axis = "C") override;
        mv::Data::TensorIterator sparsityMap(const std::string& name, const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order);
        mv::Data::TensorIterator tanh(const std::string& name, Data::TensorIterator data) override;
        mv::Data::TensorIterator tile(const std::string& name, Data::TensorIterator data, const unsigned& axis, const unsigned& tiles) override;
        mv::Data::TensorIterator topK(const std::string& name, Data::TensorIterator data, const std::string& sort, const std::string& mode, const int64_t& top_k, const int64_t& axis = 99) override;
        mv::Data::TensorIterator uPATaskDummy(const std::string& name, const std::vector< Data::TensorIterator >& inputs);
mv::Data::TensorIterator uPATaskIdentity(const std::string& name, const std::vector< Data::TensorIterator >& inputs);
mv::Data::TensorIterator uPATaskSoftmax(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& axis = "C");
mv::Data::TensorIterator uPATaskProposal(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::vector<float>& scale, const std::vector<float>& ratio, const unsigned& base_size, const unsigned& pre_nms_topn, const unsigned& post_nms_topn, const double& nms_thresh, const unsigned& feat_stride, const unsigned& min_size, const double& pre_nms_thresh = 0.000000000000000, const bool& clip_before_nms = true, const bool& clip_after_nms = false, const bool& normalize = false, const double& box_size_scale = 1.000000000000000, const double& box_coordinate_scale = 1.000000000000000, const std::string& framework = "TENSORFLOW", const bool& for_deformable = false);
mv::Data::TensorIterator uPATaskROIPooling(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& pooled_w, const unsigned& pooled_h, const double& spatial_scale, const unsigned& roi_pooling_method, const unsigned& num_rois);
mv::Data::TensorIterator uPATaskPSROIPooling(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::size_t& output_dim, const std::size_t& group_size, const double& spatial_scale, const std::size_t& pooled_w, const std::size_t& pooled_h, const std::size_t& spatial_bin_x, const std::size_t& spatial_bin_y, const std::string& mode);
mv::Data::TensorIterator uPATaskQuantize(const std::string& name, const std::vector< Data::TensorIterator >& inputs);
mv::Data::TensorIterator uPATaskReshape(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const Shape& shape);
mv::Data::TensorIterator uPATaskRegionYolo(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& coords, const unsigned& classes, const bool& do_softmax, const unsigned& num = 0, const std::vector<unsigned>& mask = {});
mv::Data::TensorIterator uPATaskReorgYolo(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& stride);
mv::Data::TensorIterator uPATaskNormalize(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const double& eps, const unsigned& across_spatial = 0, const unsigned& channel_shared = 0);
mv::Data::TensorIterator uPATaskPermute(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const Order& order);
mv::Data::TensorIterator uPATaskEltwise(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& eltwiseType);
mv::Data::TensorIterator uPATaskInterp(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const double& factor, const unsigned& pad_beg, const unsigned& pad_end, const unsigned& height = 0, const unsigned& width = 0, const bool& align_corners = true);
mv::Data::TensorIterator uPATaskDetectionOutput(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const int64_t& num_classes, const int64_t& keep_top_k, const double& nms_threshold, const int64_t& background_label_id, const int64_t& top_k, const bool& variance_encoded_in_target, const std::string& code_type, const bool& share_location, const double& confidence_threshold, const bool& clip_before_nms, const bool& clip_after_nms, const int64_t& decrease_label_id, const bool& normalized, const int64_t& input_height, const int64_t& input_width, const double& objectness_score);
mv::Data::TensorIterator uPATaskPriorbox(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& flip, const unsigned& clip, const double& step_w, const double& step_h, const double& offset);
mv::Data::TensorIterator uPATaskArgmax(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const int64_t& out_max_val, const int64_t& top_k, const int64_t& axis = 99);
mv::Data::TensorIterator uPATaskTopK(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& sort, const std::string& mode, const int64_t& top_k, const int64_t& axis = 99);
mv::Data::TensorIterator uPATaskNorm(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const double& alpha, const double& beta, const std::string& region, const unsigned& local_size);
mv::Data::TensorIterator uPATaskResample(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::string& interpolation, const bool& antialias, const Shape& output_shape);
mv::Data::TensorIterator uPATaskFakeQuantize(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& levels);
mv::Data::TensorIterator uPATaskCustom(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::vector<uint8_t>& kernelData, const std::vector<uint8_t>& paramData, const std::vector<mv::TensorInfo>& outputsInfo);
mv::Data::TensorIterator uPATaskSigmoid(const std::string& name, const std::vector< Data::TensorIterator >& inputs);
mv::Data::TensorIterator uPATaskDeconv(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1, const bool& is_depthwise = false);
mv::Data::TensorIterator uPATaskTile(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& axis, const unsigned& tiles);
mv::Data::TensorIterator uPATaskCTCDecoder(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const bool& ctc_merge_repeated);
mv::Data::TensorIterator uPATaskRefConv(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const std::array<unsigned short, 2>& stride, const std::array<unsigned short, 4>& padding, const unsigned& dilationFactor = 1, const unsigned& group = 1);
mv::Data::TensorIterator uPATaskGather(const std::string& name, const std::vector< Data::TensorIterator >& inputs, const unsigned& axis);
mv::Data::TensorIterator uPATaskHSwish(const std::string& name, const std::vector< Data::TensorIterator >& inputs);
        mv::Data::TensorIterator weightsTable(const std::string& name, const std::vector<int64_t>& data, const Shape& shape, const DType& dType, const Order& order);

        Data::OpListIterator getSourceOp(Data::TensorIterator tensor) override;
        void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr) override;
        bool isValid() const override;
        bool isValid(Data::TensorIterator tensor) const override;
        bool isValid(Data::OpListIterator op) const override;
        std::string getName() const override;

    };

}

#endif //MV_OP_MODEL_HPP_
