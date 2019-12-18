#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void convertOpsToTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void setUpPPETasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

void addPpeTask(mv::Data::OpListIterator &opIt, const std::vector<std::string> &ppeTaskType, double leakyAlpha = 0);
int32_t computeClampHigh(mv::Data::OpListIterator &opIt);
int32_t computeClampLow(mv::Data::OpListIterator &opIt);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvertOpsToTasks)
            .setFunc(convertOpsToTasksFcn)
            .setDescription(
                "Replace all operations with Hw compatible tasks.");

        MV_REGISTER_PASS(SetUpPPETasks)
            .setFunc(setUpPPETasksFcn)
            .setDescription(
                "Set up PPE Tasks for DPU Tasks");
    }
}

void setUpPPETasksFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);

    auto dpuTasks = om.getOps("DPUTask");
    for(auto& dpuTask : dpuTasks)
    {
        double leakyAlpha = 0;
        if(dpuTask->hasAttr("leakyAlpha"))
            leakyAlpha = dpuTask->get<double>("leakyAlpha");
        std::vector<std::string> postOps;
        if(dpuTask->hasAttr("postOpTypes"))
            postOps = dpuTask->get<std::vector<std::string>>("postOpTypes");
        addPpeTask(dpuTask, postOps, leakyAlpha);
    }
}

mv::Data::TensorIterator convertEltwiseToTask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software)
{
    auto eltwiseType = attrs.at("eltwiseType").get<std::string>();
    auto outputTensorType = attrs.at("dType").get<mv::DType>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    mv::Data::TensorIterator eltwiseTask;

    if (!software)
    {
        const std::array<unsigned short, 2> FAKE_KERNEL = {1,1};
        const std::array<unsigned short, 2> FAKE_STRIDE = {1,1};

        auto dpuElementWise = om.dPUTaskEltwise(inputs, eltwiseType, outputTensorType, quantParams,
                                                mv::createDPUTaskName(name));
        auto dpuElementWiseOp = om.getSourceOp(dpuElementWise);
        dpuElementWiseOp->set<std::array<unsigned short, 2>>("kSize", FAKE_KERNEL);
        dpuElementWiseOp->set<std::array<unsigned short, 2>>("stride", FAKE_STRIDE);
        dpuElementWiseOp->set<bool>("hasWeights", false);

        std::vector<std::string> postOps;
        if(dpuElementWiseOp->hasAttr("postOpTypes"))
            postOps = dpuElementWiseOp->get<std::vector<std::string>>("postOpTypes");
        postOps.push_back(eltwiseType);
        dpuElementWiseOp->set<std::vector<std::string>>("postOpTypes", postOps);
        eltwiseTask = dpuElementWise;
    }
    else
    {
        //Note: Re-write maybe DPU tasks changed them
        eltwiseTask = om.uPATaskEltwise(inputs, eltwiseType, mv::DType("Float16"), quantParams, mv::createDPUTaskName(name));
    }
    return eltwiseTask;
}


mv::Data::TensorIterator convertMaxPoolToDPUTask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto strides = attrs.at("stride").get<std::array<unsigned short, 2>>();
    auto padding = attrs.at("padding").get<std::array<unsigned short, 4>>();
    auto kernelSize = attrs.at("kSize").get<std::array<unsigned short, 2>>();
    auto exclude_pad = attrs.at("exclude_pad").get<bool>();
    auto auto_pad = attrs.at("auto_pad").get<std::string>();
    auto rounding_type = attrs.at("rounding_type").get<std::string>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto outputTensorType = attrs.at("dType").get<mv::DType>();

    auto dpuPool = om.dPUTaskMaxPool(inputs, kernelSize, strides, padding,
                       exclude_pad, auto_pad, rounding_type, outputTensorType, quantParams, mv::createDPUTaskName(name));

    om.getSourceOp(dpuPool)->set<bool>("hasWeights", false);
    return dpuPool;
}

mv::Data::TensorIterator convertDepthwiseConvolutionToDPUTask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto strides = attrs.at("stride").get<std::array<unsigned short, 2>>();
    auto padding = attrs.at("padding").get<std::array<unsigned short, 4>>();
    auto dilationFactor = attrs.at("dilationFactor").get<unsigned>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto outputTensorType = attrs.at("dType").get<mv::DType>();

    auto dpuConv = om.dPUTaskDepthwiseConv(inputs, strides, padding, dilationFactor, outputTensorType, quantParams,
                                           mv::createDPUTaskName(name));

    auto dpuConvOp = om.getSourceOp(dpuConv);
    dpuConvOp->set<bool>("hasWeights", true);

    return dpuConv;
}

mv::Data::TensorIterator convertConvolutionToDPUTask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto strides = attrs.at("stride").get<std::array<unsigned short, 2>>();
    auto padding = attrs.at("padding").get<std::array<unsigned short, 4>>();
    auto dilationFactor = attrs.at("dilationFactor").get<unsigned>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto outputTensorType = attrs.at("dType").get<mv::DType>();
    auto globalParams = om.getGlobalConfigParams();
    bool enableChannelMajor = globalParams->get<bool>("enable_channel_major_conv");

    unsigned group = attrs.at("group").get<unsigned>();

    auto dpuConv = om.dPUTaskConv(inputs, strides, padding, dilationFactor, group, outputTensorType, quantParams, mv::createDPUTaskName(name));

    auto dpuConvOp = om.getSourceOp(dpuConv);
    dpuConvOp->set<bool>("hasWeights", true);

    //    NOTE: Thanks to proper padding handling we don't need this anymore
    //    Leaving it here as an historical note... and now it's back as an option
    if(enableChannelMajor and inputs[1]->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)
    {
           dpuConvOp->erase("taskOp");
           dpuConvOp->set<std::string>("taskOp", "ChannelMajorConvolution");
    }

    return dpuConv;
}

mv::Data::TensorIterator convertIdentityToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto dtype = attrs.at("dType").get<mv::DType>();

    return om.uPATaskIdentity(inputs, dtype, quantParams, name);
}

mv::Data::TensorIterator convertSoftmaxToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto axis = attrs.at("axis").get<std::string>();
    auto dtype = attrs.at("dType").get<mv::DType>();

   return om.uPATaskSoftmax(inputs, axis, dtype, quantParams, name);
}

mv::Data::TensorIterator convertProposalToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto dtype = attrs.at("dType").get<mv::DType>();

    // Required params
    auto base_size = attrs.at("base_size").get<unsigned>();
    auto pre_nms_topn = attrs.at("pre_nms_topn").get<unsigned>();
    auto post_nms_topn = attrs.at("post_nms_topn").get<unsigned>();
    auto nms_thresh = attrs.at("nms_thresh").get<double>();
    auto feat_stride = attrs.at("feat_stride").get<unsigned>();
    auto min_size = attrs.at("min_size").get<unsigned>();

    // Optional params
    auto pre_nms_thresh = attrs.at("pre_nms_thresh").get<double>();
    auto clip_before_nms = attrs.at("clip_before_nms").get<bool>();
    auto clip_after_nms = attrs.at("clip_after_nms").get<bool>();
    auto normalize = attrs.at("normalize").get<bool>();
    auto box_size_scale = attrs.at("box_size_scale").get<double>();
    auto box_coordinate_scale = attrs.at("box_coordinate_scale").get<double>();
    auto framework = attrs.at("framework").get<std::string>();
    auto for_deformable = attrs.at("for_deformable").get<bool>();

    return om.uPATaskProposal(inputs, base_size, pre_nms_topn, post_nms_topn, nms_thresh, feat_stride, min_size,
                                                              pre_nms_thresh, clip_before_nms, clip_after_nms, normalize, box_size_scale, box_coordinate_scale, framework, for_deformable,
                                                              dtype, quantParams, name);
}

mv::Data::TensorIterator convertROIPoolingToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto dtype = attrs.at("dType").get<mv::DType>();
    auto pooled_w = attrs.at("pooled_w").get<unsigned>();
    auto pooled_h = attrs.at("pooled_h").get<unsigned>();
    auto spatial_scale = attrs.at("spatial_scale").get<double>();
    auto roi_pooling_method = attrs.at("roi_pooling_method").get<unsigned>();
    auto num_rois = attrs.at("num_rois").get<unsigned>();

    return om.uPATaskROIPooling(inputs, pooled_w, pooled_h, spatial_scale, roi_pooling_method, num_rois, dtype, quantParams, name);
}

mv::Data::TensorIterator convertQuantizeToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto dtype = attrs.at("dType").get<mv::DType>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();

    return om.uPATaskQuantize(inputs, dtype, quantParams, name);
}

mv::Data::TensorIterator convertReshapeToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto shape = attrs.at("shape").get<mv::Shape>();
    auto dtype = attrs.at("dType").get<mv::DType>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();

    return om.uPATaskReshape(inputs, shape, dtype, quantParams, name);
}

mv::Data::TensorIterator convertRegionYoloToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto coords = attrs.at("coords").get<unsigned>();
    auto classes = attrs.at("classes").get<unsigned>();
    auto do_softmax = attrs.at("do_softmax").get<bool>();
    auto num = attrs.at("num").get<unsigned>();
    auto mask = attrs.at("mask").get<std::vector<unsigned>>();
    auto dtype = attrs.at("dType").get<mv::DType>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();

    return om.uPATaskRegionYolo(inputs, coords, classes, do_softmax, num, mask, dtype, quantParams, name);
}

mv::Data::TensorIterator convertReorgYoloToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto stride = attrs.at("stride").get<unsigned>();
    auto dtype = attrs.at("dType").get<mv::DType>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();

    return om.uPATaskReorgYolo(inputs, stride, dtype, quantParams, name);
}

mv::Data::TensorIterator convertNormalizeToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto dtype = attrs.at("dType").get<mv::DType>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto eps = attrs.at("eps").get<double>();
    auto across_spatial = attrs.at("across_spatial").get<unsigned>();
    auto channel_shared = attrs.at("channel_shared").get<unsigned>();

    return om.uPATaskNormalize(inputs, eps, across_spatial, channel_shared, dtype, quantParams, name);
}

mv::Data::TensorIterator convertDetectionOutputToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs, const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto dtype = attrs.at("dType").get<mv::DType>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto num_classes = attrs.at("num_classes").get<int64_t>();
    auto keep_top_k = attrs.at("keep_top_k").get<int64_t>();
    auto nms_threshold = attrs.at("nms_threshold").get<double>();
    auto background_label_id = attrs.at("background_label_id").get<int64_t>();
    auto top_k = attrs.at("top_k").get<int64_t>();
    auto variance_encoded_in_target = attrs.at("variance_encoded_in_target").get<bool>();
    auto code_type = attrs.at("code_type").get<std::string>();
    auto share_location = attrs.at("share_location").get<bool>();
    auto confidence_threshold = attrs.at("confidence_threshold").get<double>();
    auto clip_before_nms = attrs.at("clip_before_nms").get<bool>();
    auto clip_after_nms = attrs.at("clip_after_nms").get<bool>();
    auto decrease_label_id = attrs.at("decrease_label_id").get<int64_t>();
    auto normalized = attrs.at("normalized").get<bool>();
    auto input_height = attrs.at("input_height").get<int64_t>();
    auto input_width = attrs.at("input_width").get<int64_t>();
    auto objectness_score = attrs.at("objectness_score").get<double>();

    return om.uPATaskDetectionOutput(inputs, num_classes, keep_top_k, nms_threshold, background_label_id, top_k, variance_encoded_in_target,
                                     code_type, share_location, confidence_threshold, clip_before_nms, clip_after_nms,
                                     decrease_label_id, normalized, input_height, input_width, objectness_score, dtype, quantParams);
}

mv::Data::TensorIterator convertPriorboxToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs, const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto dtype = attrs.at("dType").get<mv::DType>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto flip = attrs.at("flip").get<unsigned>();
    auto clip = attrs.at("clip").get<unsigned>();
    auto step_w = attrs.at("step_w").get<double>();
    auto step_h = attrs.at("step_h").get<double>();
    auto offset = attrs.at("offset").get<double>();

    return om.uPATaskPriorbox(inputs, flip, clip, step_w, step_h, offset, dtype, quantParams);
}

mv::Data::TensorIterator convertArgmaxToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs, const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto dtype = attrs.at("dType").get<mv::DType>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto out_max_val = attrs.at("out_max_val").get<int64_t>();
    auto top_k = attrs.at("top_k").get<int64_t>();
    auto axis = attrs.at("axis").get<int64_t>();

    return om.uPATaskArgmax(inputs, out_max_val, top_k, axis, dtype, quantParams);
}

mv::Data::TensorIterator convertPermuteToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs,
                    const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto order = attrs.at("order").get<mv::Order>();
    auto dtype = attrs.at("dType").get<mv::DType>();
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();

    mv::Data::TensorIterator upaPermute = om.uPATaskPermute(inputs, order, dtype, quantParams, name);
    auto upaPermuteOp = om.getSourceOp(upaPermute);
    // Correct order of strings if necessary
    auto old_order_str = inputs[0]->getOrder().toString();
    auto new_order_str = order.toString();

    if (old_order_str[0] != 'N')
        old_order_str = std::string(old_order_str.rbegin(), old_order_str.rend());

    if (new_order_str[0] != 'N')
        new_order_str = std::string(new_order_str.rbegin(), new_order_str.rend());

    // Convert permute order to axes
    std::vector<unsigned> permute_order(3);
    for (auto i=0; i < 3; i++)
    {
        for (auto j=0; j < 3; j++)
        {
            if (new_order_str[i+1] == old_order_str[j+1])
                permute_order.at(i) = j;
        }

    }

    upaPermuteOp->set<unsigned>("permute_order_x", permute_order.at(0));
    upaPermuteOp->set<unsigned>("permute_order_y", permute_order.at(1));
    upaPermuteOp->set<unsigned>("permute_order_z", permute_order.at(2));

    return upaPermute;
}

mv::Data::TensorIterator convertInterpToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs, const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto dtype = attrs.at("dType").get<mv::DType>();

    // Required params
    auto factor = attrs.at("factor").get<double>();
    auto pad_beg = attrs.at("pad_beg").get<unsigned>();
    auto pad_end = attrs.at("pad_end").get<unsigned>();

    // Optional params
    auto height = attrs.at("height").get<unsigned>();
    auto width = attrs.at("width").get<unsigned>();
    auto align_corners = attrs.at("align_corners").get<bool>();

    return om.uPATaskInterp(inputs, factor, pad_beg, pad_end, height, width, align_corners, dtype, quantParams, name);
}

mv::Data::TensorIterator convertNormToUPATask(mv::OpModel& om, const std::vector<mv::Data::TensorIterator>& inputs, const std::map<std::string, mv::Attribute>& attrs, const std::string& name, bool software = false)
{
    auto quantParams = attrs.at("quantParams").get<mv::QuantizationParams>();
    auto dtype = attrs.at("dType").get<mv::DType>();

    // Required params
    auto alpha = attrs.at("alpha").get<double>();
    auto beta = attrs.at("beta").get<double>();
    auto region = attrs.at("region").get<std::string>();
    auto local_size = attrs.at("local_size").get<unsigned>();

    return om.uPATaskNorm(inputs, alpha, beta, region, local_size, dtype, quantParams, name);
}

void convertOpsToTasksFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::ControlModel cm(model);
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    //Note: Eltwise might be UPA might be DPU task...
    std::vector<std::string> opsTypesToConvert = {"Conv", "DepthwiseConv", "MaxPool", "Eltwise"};
    std::vector<std::string> opsTypesToConvertToUPA = {"Argmax", "Identity", "Softmax", "Proposal", "ROIPooling",
                                                       "Quantize", "Reshape", "RegionYolo", "ReorgYolo",
                                                       "Normalize", "DetectionOutput", "Priorbox", "Permute", "Interp", "Norm"};

    opsTypesToConvert.insert(opsTypesToConvert.end(), opsTypesToConvertToUPA.begin(), opsTypesToConvertToUPA.end());
    auto opsToConvert = om.getOpsOfTypes(opsTypesToConvert);

    std::unordered_map<std::string,
            std::function<mv::Data::TensorIterator(mv::OpModel&, const std::vector<mv::Data::TensorIterator>&,
            const std::map<std::string, mv::Attribute>&, const std::string&, bool &)>> opsFunctors = {
    {"Conv", convertConvolutionToDPUTask},
    {"DepthwiseConv", convertDepthwiseConvolutionToDPUTask},
    {"MaxPool", convertMaxPoolToDPUTask},
    {"Eltwise", convertEltwiseToTask},
    {"Identity", convertIdentityToUPATask},
    {"Softmax", convertSoftmaxToUPATask},
    {"Proposal", convertProposalToUPATask},
    {"ROIPooling", convertROIPoolingToUPATask},
    {"Quantize", convertQuantizeToUPATask},
    {"Reshape", convertReshapeToUPATask},
    {"RegionYolo", convertRegionYoloToUPATask},
    {"ReorgYolo", convertReorgYoloToUPATask},
    {"Normalize", convertNormalizeToUPATask},
    {"DetectionOutput", convertDetectionOutputToUPATask},
    {"Interp", convertInterpToUPATask},
    {"Norm", convertNormToUPATask},
    {"Priorbox", convertPriorboxToUPATask},
    {"Argmax", convertArgmaxToUPATask},
    {"Permute", convertPermuteToUPATask}
    };

    bool DPUTasksinSW = globalParams->hasAttr("DPUTasksinFloat") ? globalParams->get<bool>("DPUTasksinFloat") : false;

    for(auto& opType: opsTypesToConvert)
    {
        auto ops = opsToConvert[opType];
        for(auto& opIt: ops)
        {
            bool software = false;
            //Note: That condition is coming due to limitations like add with different scales
            if (opIt->hasAttr("softwareExecuted") && opIt->get<bool>("softwareExecuted"))
                software = true;
            software = (software || DPUTasksinSW);
            auto name = opIt->getName();
            auto attrsToCopy = opIt->getAttrs();
            auto inputs = opIt->getInputTensor();
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            auto newTensor = opsFunctors[opType](om, inputs, attrsToCopy, name, software);

            newTensor->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            auto newTensorOp = om.getSourceOp(newTensor);

            if (newTensorOp->getOpType() == "DPUTask")
            {
                if (!software)
                    newTensor->set<mv::DType>("dType", mv::DType("UInt8"));
                else
                    newTensor->set<mv::DType>("dType", mv::DType("Float16"));
            }
            else if(newTensorOp->getOpType() == "UPATask") // UPA
                newTensor->set<mv::DType>("dType", mv::DType("Float16"));

            newTensorOp->setAttrs(attrsToCopy);
            setOutputDataFlow(om, newTensor, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(newTensorOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(newTensorOp), outputControlFlows);
        }
    }
}

void addPpeTask(mv::Data::OpListIterator &opIt, const std::vector<std::string>& ppeTaskTypes, double leakyAlpha)
{
    auto ppeFixedFunction = mv::PPEFixedFunction();

    ppeFixedFunction.setLowClamp(computeClampLow(opIt));
    ppeFixedFunction.setHighClamp(computeClampHigh(opIt));

    if (std::find(ppeTaskTypes.begin(), ppeTaskTypes.end(), "LeakyRelu") != ppeTaskTypes.end())
    {
        // NOTE: What are the default values here
        int8_t ppeMult;
        uint8_t ppeShift;
        if (leakyAlpha != 0)
        {
            // HW PRELU MULT is I8, so 7 precision bits are available
            unsigned bits = 7;
            int exponent;
            double mantissa;

            mantissa = std::frexp(leakyAlpha, &exponent);
            ppeShift = bits - exponent;
            ppeMult = (mantissa * pow(2, bits));
        }
        ppeFixedFunction.setLReluMult(ppeMult);
        ppeFixedFunction.setLReluShift(ppeShift);
    }

    for(auto& ppeTaskType: ppeTaskTypes)
    {
        auto ppeLayerType = mv::PPELayerType(ppeTaskType);
        ppeFixedFunction.addLayer(ppeLayerType);
    }

    auto ppeTask = mv::PPETask(ppeFixedFunction);
    opIt->set<mv::PPETask>("PPETask", ppeTask);
}


// CLAMP FUNCTIONS FROM HERE

// ASSUMPTION:
// A model can give us clamp values either in I32 or FP32 or not clamp at all.
// The assumption is that in any case this clamp value IS NOT quantized.


// When outputDType is U8 or I8 we have to compute saturation clamp in every case
// U8 <- [-zp; 255 - zp]
// I8 <- [-128 - zp; +127 - zp] but ensure that zp is 0
// The clamp value is stored as is if compute type is U8. Otherwise it must be converted in S16.16
int32_t computeClampLow(mv::Data::OpListIterator &opIt)
{
    auto computeDType = opIt->getInputTensor(0)->getDType();
    auto outputDType = opIt->getOutputTensor(0)->getDType();
    auto U8 = mv::DType("UInt8");
    auto I8 = mv::DType("Int8");
    auto FP16 = mv::DType("Float16");

    int32_t clamp = -2147483648;

    if (outputDType == U8 || outputDType == I8)
    {
        // Saturation clamp has to be computed in this case
        mv::QuantizationParams outputQuantParams = opIt->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        int32_t saturationClamp = - outputQuantParams.getZeroPoint()[0];
        if(outputDType == I8)
            saturationClamp -= 128;

        clamp = saturationClamp;

        if(opIt->hasAttr("Maximum"))
        {
            double clampValue = opIt->get<double>("Maximum");
            double outputScale = outputQuantParams.getScale()[0];
            int32_t quantizedClampValue = static_cast<int32_t>(clampValue / outputScale);

            if(quantizedClampValue > clamp)
                clamp = quantizedClampValue;
        }

        if(computeDType == FP16)
            clamp <<= 16;
    }

    else if (outputDType == FP16)
    {
        if(opIt->hasAttr("Maximum"))
        {
            double clampValue = opIt->get<double>("Maximum");

            if(computeDType == U8 || computeDType == I8)
                clamp = static_cast<int32_t>(clampValue);
            else if (computeDType == FP16)
                clamp = static_cast<int32_t>(clampValue * pow(2,16));
        }
    }
    return clamp;
}


int32_t computeClampHigh(mv::Data::OpListIterator &opIt)
{
    auto computeDType = opIt->getInputTensor(0)->getDType();
    auto outputDType = opIt->getOutputTensor(0)->getDType();
    auto U8 = mv::DType("UInt8");
    auto I8 = mv::DType("Int8");
    auto FP16 = mv::DType("Float16");

    int32_t clamp = 2147483647;

    if (outputDType == U8 || outputDType == I8)
    {
        // Saturation clamp has to be computed in this case
        mv::QuantizationParams outputQuantParams = opIt->getOutputTensor(0)->get<mv::QuantizationParams>("quantParams");
        int32_t saturationClamp = - outputQuantParams.getZeroPoint()[0];
        if(outputDType == I8)
            saturationClamp += 127;
        else
            saturationClamp += 255;

        clamp = saturationClamp;

        if(opIt->hasAttr("Minimum"))
        {
            double clampValue = opIt->get<double>("Minimum");
            double outputScale = outputQuantParams.getScale()[0];
            int32_t quantizedClampValue = static_cast<int32_t>(clampValue / outputScale);

            if(quantizedClampValue < clamp)
                clamp = quantizedClampValue;
        }

        if(computeDType == FP16)
            clamp <<= 16;
    }

    else if (outputDType == FP16)
    {
        if(opIt->hasAttr("Minimum"))
        {
            double clampValue = opIt->get<double>("Minimum");

            if(computeDType == U8 || computeDType == I8)
                clamp = static_cast<int32_t>(clampValue);
            else if (computeDType == FP16)
                clamp = static_cast<int32_t>(clampValue * pow(2,16));
        }
    }
    return clamp;
}
