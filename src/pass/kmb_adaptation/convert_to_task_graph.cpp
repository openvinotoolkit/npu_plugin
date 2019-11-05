#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static void convertOpsToDPUTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void convertOpsToUPATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void setUpPPETasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

void addPpeTask(mv::Data::OpListIterator &opIt, const std::vector<std::string> &ppeTaskType, double leakyAlpha = 0);
int32_t computeMaxClampValue(mv::Data::OpListIterator &opIt);
int32_t computeMinClampValue(mv::Data::OpListIterator &opIt);
std::pair<int32_t, int32_t> computeClampValues(mv::Data::OpListIterator &opIt);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvertOpsToDPUTasks)
            .setFunc(convertOpsToDPUTasksFcn)
            .setDescription(
                "Replace all convolution operations with DPU tasks.\n"
                "Assume each convolution can be done with DPU on KMB.\n"
                "Assume each convolution should be done on DPU.");


        MV_REGISTER_PASS(ConvertOpsToUPATasks)
            .setFunc(convertOpsToUPATasksFcn)
            .setDescription(
                "Replace all supported operations with UPA tasks.");

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

void convertOpsToDPUTasksFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    const std::array<unsigned short, 2> FAKE_KERNEL = {1,1};
    const std::array<unsigned short, 2> FAKE_STRIDE = {1,1};

    // Pass main assumption is that we are working on the original graph (just AveragePooling substituted)

    // While loop is preferred in a loop like this were we are performing eliminations
    // as it gives more flexibility on when to increment the iterator
    auto opIt = om.getInput();
    while (opIt != om.opEnd())
    {
        std::string opType = opIt->getOpType();

        if (opType == "Conv" || opType == "DepthwiseConv")
        {
            auto input = opIt->getInputTensor(0);
            auto kernel = opIt->getInputTensor(1);

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");
            opIt->set<bool>("hasWeights", true);

            auto name = opIt->getName();
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
            auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

            unsigned group = 1;
            if (opType == "Conv")
                group = opIt->get<unsigned>("group");
            auto attrsToCopy = opIt->getAttrs();

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator dpuConv;
            if(opType == "Conv")
                dpuConv = om.dPUTaskConv({input, kernel}, strides, padding, dilationFactor, group, outputTensorType, quantParams, mv::createDPUTaskName(name));
            else
                dpuConv = om.dPUTaskDepthwiseConv({input, kernel}, strides, padding, dilationFactor, outputTensorType, quantParams, mv::createDPUTaskName(name));

            dpuConv->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            auto dpuConvOp = om.getSourceOp(dpuConv);

            dpuConvOp->setAttrs(attrsToCopy);
            setOutputDataFlow(om, dpuConv, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(dpuConvOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(dpuConvOp), outputControlFlows);

            // NOTE: If we want to get rid of ChannelMajorConvolution we have to act here
            if(opType == "Conv")
            {
                if(kernel->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)
                {
                    dpuConvOp->erase("taskOp");
                    dpuConvOp->set<std::string>("taskOp", "ChannelMajorConvolution");
                }
            }

        }
        else if (opType == "MaxPool")
        {
            auto input = opIt->getInputTensor(0);

            opIt->set<bool>("hasWeights", false);

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto kernelSize = opIt->get<std::array<unsigned short, 2>>("kSize");
            auto exclude_pad = opIt->get<bool>("exclude_pad");
            auto auto_pad = opIt->get<std::string>("auto_pad");
            auto rounding_type = opIt->get<std::string>("rounding_type");
            auto name = opIt->getName();
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
            auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");
            auto attrsToCopy = opIt->getAttrs();
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            auto dpuPool = om.dPUTaskMaxPool({input}, kernelSize, strides, padding,
                               exclude_pad, auto_pad, rounding_type, outputTensorType, quantParams, mv::createDPUTaskName(name));
            auto dpuPoolOp = om.getSourceOp(dpuPool);
            dpuPool->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);

            dpuPoolOp->setAttrs(attrsToCopy);
            setOutputDataFlow(om, dpuPool, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(dpuPoolOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(dpuPoolOp), outputControlFlows);
        }
        else if (opType == "Eltwise")
        {
            auto eltwiseType = opIt->get<std::string>("eltwiseType");
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");
            opIt->set<bool>("hasWeights", false);

            auto input1 = opIt->getInputTensor(0);
            auto input2 = opIt->getInputTensor(1);
            std::vector<mv::Data::TensorIterator> inputs;
            inputs.push_back(input1);
            inputs.push_back(input2);
            auto name = opIt->getName();
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");

            auto attrsToCopy = opIt->getAttrs();
            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            auto dpuElementWise = om.dPUTaskEltwise(inputs, eltwiseType, outputTensorType, quantParams, mv::createDPUTaskName(name));
            dpuElementWise->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);

            auto dpuElementWiseOp = om.getSourceOp(dpuElementWise);
            dpuElementWiseOp->setAttrs(attrsToCopy);

            dpuElementWiseOp->set<std::array<unsigned short, 2>>("kSize", FAKE_KERNEL);
            dpuElementWiseOp->set<std::array<unsigned short, 2>>("stride", FAKE_STRIDE);

            mv::setOutputDataFlow(om, dpuElementWise, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(dpuElementWiseOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(dpuElementWiseOp), outputControlFlows);
        }
        else
            ++opIt;
    }
}

void convertOpsToUPATasksFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    // Pass main assumption is that we are working on the original graph (just AveragePooling substituted)

    // While loop is preferred in a loop like this were we are performing eliminations
    // as it gives more flexibility on when to increment the iterator
    auto opIt = om.getInput();
    while (opIt != om.opEnd())
    {
        std::string opType = opIt->getOpType();

        if (opType == "Identity")
        {
            auto input = opIt->getInputTensor(0);
            auto output = opIt->getOutputTensor(0);
            auto outputMemoryLocation = output->get<mv::Tensor::MemoryLocation>("Location");
            auto splitStrategy = opIt->get<std::string>("splitStrategy");

            unsigned opId = opIt->get<unsigned>("opId");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator upaTask = om.uPATaskIdentity({input});

            auto upaTaskOp = om.getSourceOp(upaTask);
            upaTaskOp->set<unsigned>("opId", opId);

            upaTask->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            upaTaskOp->set<std::string>("splitStrategy", splitStrategy);

            setOutputDataFlow(om, upaTask, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaTaskOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaTaskOp), outputControlFlows);
        }
        else if (opType == "Dummy")
        {
            auto input = opIt->getInputTensor(0);
            mv::getOutputDataFlow(om, opIt);
            mv::Data::TensorIterator upaTask = om.uPATaskDummy({input});
        }
        else if (opType == "Softmax")
        {
            auto input = opIt->getInputTensor(0);
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            unsigned opId = opIt->get<unsigned>("opId");
            auto dtype = opIt->get<mv::DType>("dType");
            auto splitStrategy = opIt->get<std::string>("splitStrategy");

            auto axis = opIt->get<std::string>("axis");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator upaTask = om.uPATaskSoftmax({input}, axis, dtype);

            auto upaTaskOp = om.getSourceOp(upaTask);
            upaTaskOp->set<unsigned>("opId", opId);
            upaTaskOp->set<std::string>("splitStrategy", splitStrategy);
            upaTask->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            setOutputDataFlow(om, upaTask, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaTaskOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaTaskOp), outputControlFlows);
        }
        else if (opType == "Proposal")
        {
            auto cls_pred = opIt->getInputTensor(0);
            auto bbox_pred = opIt->getInputTensor(1);
            auto im_info = opIt->getInputTensor(2);
            auto scale = opIt->getInputTensor(3);
            auto ratio = opIt->getInputTensor(4);
            cls_pred->set<std::string>("populatedTensorType", "cls_pred");
            bbox_pred->set<std::string>("populatedTensorType", "bbox_pred");
            auto splitStrategy = opIt->get<std::string>("splitStrategy");

            std::vector<mv::Data::TensorIterator> inputs;
            inputs.push_back(cls_pred);
            inputs.push_back(bbox_pred);
            inputs.push_back(im_info);
            inputs.push_back(scale);
            inputs.push_back(ratio);

            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");

            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            unsigned opId = opIt->get<unsigned>("opId");
            auto dtype = opIt->get<mv::DType>("dType");

            // Required params
            auto base_size = opIt->get<unsigned>("base_size");
            auto pre_nms_topn = opIt->get<unsigned>("pre_nms_topn");
            auto post_nms_topn = opIt->get<unsigned>("post_nms_topn");
            auto nms_thresh = opIt->get<double>("nms_thresh");
            auto feat_stride = opIt->get<unsigned>("feat_stride");
            auto min_size = opIt->get<unsigned>("min_size");

            // Optional params
            auto pre_nms_thresh = opIt->get<double>("pre_nms_thresh");
            auto clip_before_nms = opIt->get<bool>("clip_before_nms");
            auto clip_after_nms = opIt->get<bool>("clip_after_nms");
            auto normalize = opIt->get<bool>("normalize");
            auto box_size_scale = opIt->get<double>("box_size_scale");
            auto box_coordinate_scale = opIt->get<double>("box_coordinate_scale");
            auto framework = opIt->get<std::string>("framework");
            auto for_deformable = opIt->get<bool>("for_deformable");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator upaProposal = om.uPATaskProposal(inputs, base_size, pre_nms_topn, post_nms_topn, nms_thresh, feat_stride, min_size,
                                                                      pre_nms_thresh, clip_before_nms, clip_after_nms, normalize, box_size_scale, box_coordinate_scale, framework, for_deformable,
                                                                      dtype, quantParams);

            auto upaProposalOp = om.getSourceOp(upaProposal);
            upaProposalOp->set<std::string>("splitStrategy", splitStrategy);
            upaProposalOp->set<unsigned>("opId", opId);

            upaProposal->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            // Required params
            upaProposal->set<unsigned>("base_size", base_size);
            upaProposal->set<unsigned>("pre_nms_topn", pre_nms_topn);
            upaProposal->set<unsigned>("post_nms_topn", post_nms_topn);
            upaProposal->set<double>("nms_thresh", nms_thresh);
            upaProposal->set<unsigned>("feat_stride", feat_stride);
            upaProposal->set<unsigned>("min_size", min_size);

            // Optional params
            upaProposal->set<double>("pre_nms_thresh", pre_nms_thresh);
            upaProposal->set<bool>("clip_before_nms", clip_before_nms);
            upaProposal->set<bool>("clip_after_nms", clip_after_nms);
            upaProposal->set<bool>("normalize", normalize);
            upaProposal->set<double>("box_size_scale", box_size_scale);
            upaProposal->set<double>("box_coordinate_scale", box_coordinate_scale);
            upaProposal->set<std::string>("framework", framework);
            upaProposal->set<bool>("for_deformable", for_deformable);

            setOutputDataFlow(om, upaProposal, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaProposalOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaProposalOp), outputControlFlows);

        }
        else if (opType == "ROIPooling")
        {
            auto input = opIt->getInputTensor(0);
            auto coords = opIt->getInputTensor(1);
            input->set<std::string>("populatedTensorType", "input");
            coords->set<std::string>("populatedTensorType", "coords");
            auto splitStrategy = opIt->get<std::string>("splitStrategy");

            std::vector<mv::Data::TensorIterator> inputs;
            inputs.push_back(input);
            inputs.push_back(coords);

            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");

            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            unsigned opId = opIt->get<unsigned>("opId");
            auto dtype = opIt->get<mv::DType>("dType");

            auto pooled_w = opIt->get<unsigned>("pooled_w");
            auto pooled_h = opIt->get<unsigned>("pooled_h");
            auto spatial_scale = opIt->get<double>("spatial_scale");
            auto roi_pooling_method = opIt->get<unsigned>("roi_pooling_method");
            auto num_rois = opIt->get<unsigned>("num_rois");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator upaROIPooling = om.uPATaskROIPooling(inputs, pooled_w, pooled_h, spatial_scale, roi_pooling_method, num_rois, dtype, quantParams);

            auto upaROIPoolingOp = om.getSourceOp(upaROIPooling);
            upaROIPoolingOp->set<std::string>("splitStrategy", splitStrategy);

            upaROIPoolingOp->set<unsigned>("opId", opId);

            upaROIPooling->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);

            upaROIPooling->set<unsigned>("pooled_w", pooled_w);
            upaROIPooling->set<unsigned>("pooled_h", pooled_h);
            upaROIPooling->set<double>("spatial_scale", spatial_scale);
            upaROIPooling->set<unsigned>("roi_pooling_method", roi_pooling_method);
            upaROIPooling->set<unsigned>("num_rois", num_rois);

            setOutputDataFlow(om, upaROIPooling, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaROIPoolingOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaROIPoolingOp), outputControlFlows);

        }
        else if (opType == "Quantize")
        {
            auto input = opIt->getInputTensor(0);
            auto output = opIt->getOutputTensor(0);
            auto outputMemoryLocation = output->get<mv::Tensor::MemoryLocation>("Location");
            unsigned opId = opIt->get<unsigned>("opId");
            auto dtype = opIt->get<mv::DType>("dType");
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
            auto splitStrategy = opIt->get<std::string>("splitStrategy");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator upaQuantize = om.uPATaskQuantize({input}, dtype, quantParams);

            auto upaQuantizeOp = om.getSourceOp(upaQuantize);
            upaQuantizeOp->set<unsigned>("opId", opId);
            upaQuantizeOp->set<std::string>("splitStrategy", splitStrategy);

            upaQuantize->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            setOutputDataFlow(om, upaQuantize, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaQuantizeOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaQuantizeOp), outputControlFlows);

        }
        else if (opType == "Reshape")
        {
            auto input = opIt->getInputTensor(0);
            auto output = opIt->getOutputTensor(0);
            auto outputMemoryLocation = output->get<mv::Tensor::MemoryLocation>("Location");
            unsigned opId = opIt->get<unsigned>("opId");
            auto shape = opIt->get<mv::Shape>("shape");
            auto order = opIt->get<mv::Order>("order");
            auto dtype = opIt->get<mv::DType>("dType");
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
            auto splitStrategy = opIt->get<std::string>("splitStrategy");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator upaReshape = om.uPATaskReshape({input}, shape, order, dtype, quantParams);

            auto upaReshapeOp = om.getSourceOp(upaReshape);
            upaReshapeOp->set<unsigned>("opId", opId);
            upaReshapeOp->set<std::string>("splitStrategy", splitStrategy);

            upaReshape->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            setOutputDataFlow(om, upaReshape, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaReshapeOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaReshapeOp), outputControlFlows);

        }
        else if (opType == "RegionYolo")
        {
            auto input = opIt->getInputTensor(0);
            auto output = opIt->getOutputTensor(0);
            auto outputMemoryLocation = output->get<mv::Tensor::MemoryLocation>("Location");
            unsigned opId = opIt->get<unsigned>("opId");
            auto coords = opIt->get<unsigned>("coords");
            auto classes = opIt->get<unsigned>("classes");
            auto do_softmax = opIt->get<bool>("do_softmax");
            auto num = opIt->get<unsigned>("num");
            auto mask = opIt->get<std::vector<unsigned>>("mask");
            auto dtype = opIt->get<mv::DType>("dType");
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
            auto splitStrategy = opIt->get<std::string>("splitStrategy");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator upaRegionYolo = om.uPATaskRegionYolo({input}, coords, classes, do_softmax, num, mask, dtype, quantParams);

            auto upaRegionYoloOp = om.getSourceOp(upaRegionYolo);
            upaRegionYoloOp->set<unsigned>("opId", opId);
            upaRegionYoloOp->set<std::string>("splitStrategy", splitStrategy);

            upaRegionYolo->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            setOutputDataFlow(om, upaRegionYolo, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaRegionYoloOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaRegionYoloOp), outputControlFlows);

        }
        else if (opType == "ReorgYolo")
        {
            auto input = opIt->getInputTensor(0);
            auto output = opIt->getOutputTensor(0);
            auto outputMemoryLocation = output->get<mv::Tensor::MemoryLocation>("Location");
            unsigned opId = opIt->get<unsigned>("opId");
            auto stride = opIt->get<unsigned>("stride");
            auto dtype = opIt->get<mv::DType>("dType");
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
            auto splitStrategy = opIt->get<std::string>("splitStrategy");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator upaReorgYolo = om.uPATaskReorgYolo({input}, stride, dtype, quantParams);

            auto upaReorgYoloOp = om.getSourceOp(upaReorgYolo);
            upaReorgYoloOp->set<unsigned>("opId", opId);
            upaReorgYoloOp->set<std::string>("splitStrategy", splitStrategy);

            upaReorgYolo->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            setOutputDataFlow(om, upaReorgYolo, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaReorgYoloOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaReorgYoloOp), outputControlFlows);
        }
        else if (opType == "Normalize")
        {
            auto input1 = opIt->getInputTensor(0);
            auto weights = opIt->getInputTensor(1);
            std::vector<mv::Data::TensorIterator> inputs;
            inputs.push_back(input1);
            inputs.push_back(weights);
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            unsigned opId = opIt->get<unsigned>("opId");
            auto dtype = opIt->get<mv::DType>("dType");
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
            auto splitStrategy = opIt->get<std::string>("splitStrategy");

            auto eps = opIt->get<double>("eps");
            unsigned across_spatial = opIt->get<unsigned>("across_spatial");
            unsigned channel_shared = opIt->get<unsigned>("channel_shared");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator upaNormalize = om.uPATaskNormalize(inputs, eps, across_spatial, channel_shared, dtype, quantParams);

            auto upaNormalizeOp = om.getSourceOp(upaNormalize);
            upaNormalizeOp->set<unsigned>("opId", opId);
            upaNormalizeOp->set<std::string>("splitStrategy", splitStrategy);

            upaNormalize->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            setOutputDataFlow(om, upaNormalize, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaNormalizeOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaNormalizeOp), outputControlFlows);

        }
        else if (opType == "Permute")
        {
            auto input = opIt->getInputTensor(0);
            auto output = opIt->getOutputTensor(0);
            auto outputMemoryLocation = output->get<mv::Tensor::MemoryLocation>("Location");
            unsigned opId = opIt->get<unsigned>("opId");
            auto order = opIt->get<mv::Order>("order");
            auto dtype = opIt->get<mv::DType>("dType");
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");
            auto splitStrategy = opIt->get<std::string>("splitStrategy");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator upaPermute = om.uPATaskPermute({input}, order, dtype, quantParams);

            auto upaPermuteOp = om.getSourceOp(upaPermute);
            upaPermuteOp->set<unsigned>("opId", opId);
            upaPermuteOp->set<std::string>("splitStrategy", splitStrategy);

            upaPermute->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            setOutputDataFlow(om, upaPermute, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaPermuteOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaPermuteOp), outputControlFlows);

        }
        else
            ++opIt;
    }
}

void addPpeTask(mv::Data::OpListIterator &opIt, const std::vector<std::string>& ppeTaskTypes, double leakyAlpha)
{
    auto ppeFixedFunction = mv::PPEFixedFunction();

    std::pair<int32_t, int32_t> clampValues = computeClampValues(opIt);
    ppeFixedFunction.setLowClamp(clampValues.first);
    ppeFixedFunction.setHighClamp(clampValues.second);

    if (std::find(ppeTaskTypes.begin(), ppeTaskTypes.end(), "LPRELU") != ppeTaskTypes.end())
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

std::pair<int32_t, int32_t> computeClampValues(mv::Data::OpListIterator &opIt)
{
    std::pair<int32_t, int32_t> toReturn;
    toReturn.first = computeMinClampValue(opIt);
    toReturn.second = computeMaxClampValue(opIt);
    return toReturn;
}

// CLAMP FUNCTIONS FROM HERE

// ASSUMPTION:
// A model can give us clamp values either in I32 or FP32 or not clamp at all.
// If the operation output is U8/I8 we assume U8 clamp
// If the operation output is FP16 we asssume double precision clamp.


// When outputDType is U8 or I8 we have to compute saturation clamp in every case
// U8 <- [-zp; 255 - zp]
// I8 <- [-128 - zp; +127 - zp] but ensure that zp is 0
// The clamp value is stored as is if compute type is U8. Otherwise it must be converted in S16.16
int32_t computeMinClampValue(mv::Data::OpListIterator &opIt)
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

        if(opIt->hasAttr("Minimum"))
        {
            double clampValue = opIt->get<int32_t>("Minimum");
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


int32_t computeMaxClampValue(mv::Data::OpListIterator &opIt)
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

        if(opIt->hasAttr("Maximum"))
        {
            double clampValue = opIt->get<int32_t>("Maximum");
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
