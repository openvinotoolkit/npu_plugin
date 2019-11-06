#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/kmb/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static const std::array<unsigned short, 2> FAKE_KERNEL = {1,1};
static const std::array<unsigned short, 2> FAKE_STRIDE = {1,1};

static void convertOpsToDPUTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void convertOpsToUPATasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void addPpeTask(mv::Data::OpListIterator &opIt, std::string ppeTaskType, double leakyAlpha = 0);
static void computeClampValues(mv::Data::OpListIterator &opIt);
union fixed_rep
{
    int32_t int_rep;
    float float_rep;
};
static void computeClampMinimum(mv::Data::OpListIterator &opIt, fixed_rep &fixed_min);
static void computeClampMaximum(mv::Data::OpListIterator &opIt, fixed_rep &fixed_max);

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
    }
}

void convertOpsToDPUTasksFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{

    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::ControlModel cm(model);

    auto addFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const std::string& s){ return om.dPUTaskAdd(vec, mv::DType("Default"), quantParams,s);};
    auto subFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const std::string& s){ return om.dPUTaskSubtract(vec, mv::DType("Default"), quantParams,s);};
    auto multFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const std::string& s){ return om.dPUTaskMultiply(vec, mv::DType("Default"), quantParams,s);};

    auto dpuTaskMap = std::map<std::string, std::function<mv::Data::TensorIterator (std::vector< mv::Data::TensorIterator >&, const mv::QuantizationParams&, const std::string&)>>
                                               {{"Add", addFcn},
                                               {"Subtract", subFcn},
                                               {"Multiply", multFcn}};
    // Pass main assumption is that we are working on the original graph (just AveragePooling substituted)

    // While loop is preferred in a loop like this were we are performing eliminations
    // as it gives more flexibility on when to increment the iterator
    auto opIt = om.getInput();
    while (opIt != om.opEnd())
    {
        std::string opType = opIt->getOpType();

        if (opType == "Conv" || opType == "DepthwiseConv")
        {
            auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

            auto input = opIt->getInputTensor(0);
            auto kernel = opIt->getInputTensor(1);

            kernel->set<std::string>("populatedTensorType", "weights");

            auto opId = opIt->get<unsigned>("opId");

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");

            auto name = opIt->getName();
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");

            std::string biasName, splitStrategy, workloadStrategyMPEMode, ppeType;
            int workloadStrategyNWorkloads = -1;

            bool inputActivationSparsity, outputActivationSparsity, weightsSparsity = false;
            //NOTE: LEAKY RELU PARAM
            double leakyAlpha = 0;
            int32_t minimum = std::numeric_limits<int32_t>::min();
            int32_t maximum = std::numeric_limits<int32_t>::max();
            std::vector <std::string> postOpTypes = {};
            unsigned group = 1;
            if (opType == "Conv")
                group = opIt->get<unsigned>("group");

            if (opIt->hasAttr("bias"))
                biasName = opIt->get<std::string>("bias");

            if(opIt->hasAttr("splitStrategy"))
                splitStrategy = opIt->get<std::string>("splitStrategy");

            if(opIt->hasAttr("inputActivationSparsity"))
                inputActivationSparsity = opIt->get<bool>("inputActivationSparsity");

            if(opIt->hasAttr("outputActivationSparsity"))
                outputActivationSparsity = opIt->get<bool>("outputActivationSparsity");

            if(opIt->hasAttr("weightsSparsity"))
                weightsSparsity = opIt->get<bool>("weightsSparsity");

            if (opIt->hasAttr("WorkloadStrategy_nWorkloads"))
                workloadStrategyMPEMode = opIt->get<std::string>("WorkloadStrategy_MPE_mode");

            if (opIt->hasAttr("WorkloadStrategy_nWorkloads"))
                workloadStrategyNWorkloads = opIt->get<int>("WorkloadStrategy_nWorkloads");
            //NOTE: This will become bigger with new ppeTasks
            if (opIt->hasAttr("postOpType"))
            {
                if (opIt->get<std::string>("postOpType") == "LeakyRelu")
                {
                    ppeType = "LPRELU";
                    leakyAlpha = opIt->get<double>("alpha");
                }
                else if (opIt->get<std::string>("postOpType") == "Relu")
                {
                    ppeType = "RELU";
                }
                else if (opIt->get<std::string>("postOpType") == "Sigmoid")
                {
                    ppeType = "SIGMOID";
                }
            }
            else if (opIt->hasAttr("postOpTypes"))
            {
                postOpTypes = opIt->get<std::vector<std::string>>("postOpTypes");
                computeClampValues(opIt);
                if (std::find(postOpTypes.begin(), postOpTypes.end(), "Minimum") != postOpTypes.end())
                    minimum = opIt->get<int32_t>("clampMinimum");
                if (std::find(postOpTypes.begin(), postOpTypes.end(), "Maximum") != postOpTypes.end())
                    maximum = opIt->get<int32_t>("clampMaximum");
            }

            std::array<unsigned short, 2> kernelSize = {kernel->getShape()[mv::KERNEL_WIDTH], kernel->getShape()[mv::KERNEL_HEIGHT]};

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator dpuConv;
            if(opType == "Conv")
                dpuConv = om.dPUTaskConv({input, kernel}, strides, padding, dilationFactor, group, mv::DType("Default"), quantParams, mv::createDPUTaskName(name));
            else
                dpuConv = om.dPUTaskDepthwiseConv({input, kernel}, strides, padding, dilationFactor, mv::DType("Default"), quantParams, mv::createDPUTaskName(name));

            auto dpuConvOp = om.getSourceOp(dpuConv);
            dpuConvOp->set<unsigned>("opId", opId);

            if(opType == "Conv")
                dpuConvOp->set<bool>("inputActivationSparsity", inputActivationSparsity);
            else
                dpuConvOp->set<bool>("inputActivationSparsity", false);
            dpuConvOp->set<bool>("outputActivationSparsity", outputActivationSparsity);
            dpuConvOp->set<bool>("weightsSparsity", weightsSparsity);

            dpuConvOp->set<bool>("hasWeights", true);
            dpuConvOp->set<std::array<unsigned short, 2>>("kSize", kernelSize);

            if (!postOpTypes.empty())
            {
                auto ppeFixedFunction = mv::PPEFixedFunction(1, 0, minimum, maximum);
                if (std::find(postOpTypes.begin(), postOpTypes.end(), "Minimum") != postOpTypes.end())
                {
                    std::string ppeLayerType0 = "MINIMUM";
                    ppeFixedFunction.addLayer(ppeLayerType0);
                }
                if (std::find(postOpTypes.begin(), postOpTypes.end(), "Maximum") != postOpTypes.end())
                {
                    std::string ppeLayerType1 = "MAXIMUM";
                    ppeFixedFunction.addLayer(ppeLayerType1);
                }
                auto ppeTask = mv::PPETask(ppeFixedFunction);
                dpuConvOp->set<mv::PPETask>("PPETask", ppeTask);
            }
            else if (!ppeType.empty())
                addPpeTask(dpuConvOp, ppeType, leakyAlpha);

            if(!biasName.empty())
               dpuConvOp->set<std::string>("bias", biasName);
            if(!splitStrategy.empty())
            {
                //NOTE:Convolution can not be HWSwitch
                dpuConvOp->set<std::string>("splitStrategy", splitStrategy);
                if (splitStrategy == "SplitOverK")
                    dpuConvOp->set<bool>("multiCast", true);
                else
                    dpuConvOp->set<bool>("multiCast", false);
            }
            if(!workloadStrategyMPEMode.empty())
                dpuConvOp->set<std::string>("WorkloadStrategy_MPE_mode", workloadStrategyMPEMode);
            if(workloadStrategyNWorkloads != -1)
                dpuConvOp->set<int>("WorkloadStrategy_nWorkloads", workloadStrategyNWorkloads);

            dpuConv->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            dpuConv->set<mv::DType>("dType", outputTensorType);
            setOutputDataFlow(om, dpuConv, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(dpuConvOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(dpuConvOp), outputControlFlows);


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
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");

            auto input = opIt->getInputTensor(0);
            auto opId = opIt->get<unsigned>("opId");

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto kernelSize = opIt->get<std::array<unsigned short, 2>>("kSize");
            auto exclude_pad = opIt->get<bool>("exclude_pad");
            auto auto_pad = opIt->get<std::string>("auto_pad");
            auto rounding_type = opIt->get<std::string>("rounding_type");
            auto name = opIt->getName();
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");

            bool outputActivationSparsity = false;

            std::string splitStrategy;
            if(opIt->hasAttr("splitStrategy"))
                splitStrategy = opIt->get<std::string>("splitStrategy");

            if(opIt->hasAttr("outputActivationSparsity"))
                outputActivationSparsity = opIt->get<bool>("outputActivationSparsity");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            auto dpuPool = om.dPUTaskMaxPool({input}, kernelSize, strides, padding,
                               exclude_pad, auto_pad, rounding_type, mv::DType("Default"), quantParams, mv::createDPUTaskName(name));
            auto dpuPoolOp = om.getSourceOp(dpuPool);
            dpuPoolOp->set<unsigned>("opId", opId);
            dpuPoolOp->set<bool>("hasWeights", false);

            dpuPoolOp->set<bool>("inputActivationSparsity", false);
            dpuPoolOp->set<bool>("outputActivationSparsity", outputActivationSparsity);
            dpuPoolOp->set<bool>("weightsSparsity", false);

            if(!splitStrategy.empty())
            {
                //NOTE:Pooling can not be SplitOverK
               dpuPoolOp->set<std::string>("splitStrategy", splitStrategy);
               if (splitStrategy == "HKSwitch")
                    dpuPoolOp->set<bool>("multiCast", true);
                else
                   dpuPoolOp->set<bool>("multiCast", false);
            }

            dpuPool->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            dpuPool->set<mv::DType>("dType", outputTensorType);
            setOutputDataFlow(om, dpuPool, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(dpuPoolOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(dpuPoolOp), outputControlFlows);
        }
        else if (opType == "Add" || opType == "Subtract" || opType == "Multiply")
        {
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
            auto outputTensorType = opIt->getOutputTensor(0)->get<mv::DType>("dType");

            auto input1 = opIt->getInputTensor(0);
            auto input2 = opIt->getInputTensor(1);
            std::vector<mv::Data::TensorIterator> inputs;
            inputs.push_back(input1);
            inputs.push_back(input2);
            auto name = opIt->getName();

            bool outputActivationSparsity = false;

            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");

            auto opId = opIt->get<unsigned>("opId");

            std::string splitStrategy;

            if(opIt->hasAttr("splitStrategy"))
                splitStrategy = opIt->get<std::string>("splitStrategy");

            if(opIt->hasAttr("outputActivationSparsity"))
                outputActivationSparsity = opIt->get<bool>("outputActivationSparsity");

            auto inputControlFlows = mv::getInputControlFlow(cm, cm.switchContext(opIt));
            auto outputControlFlows = mv::getOutputControlFlow(cm, cm.switchContext(opIt));
            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            auto dpuElementWiseFunctor = (dpuTaskMap.at(opType));
            auto dpuElementWise = dpuElementWiseFunctor(inputs, quantParams, mv::createDPUTaskName(name));
            auto dpuElementWiseOp = om.getSourceOp(dpuElementWise);
            dpuElementWiseOp->set<unsigned>("opId", opId);
            dpuElementWiseOp->set<bool>("hasWeights", false);
            dpuElementWiseOp->set<std::array<unsigned short, 2>>("kSize", FAKE_KERNEL);
            dpuElementWiseOp->set<std::array<unsigned short, 2>>("stride", FAKE_STRIDE);
            addPpeTask(dpuElementWiseOp, opType);

            // NOTE/TODO: If and when we want to support elementwise IDU sparsity we have to act here
            dpuElementWiseOp->set<bool>("inputActivationSparsity", false);
            dpuElementWiseOp->set<bool>("weightsSparsity", false);
            dpuElementWiseOp->set<bool>("outputActivationSparsity", outputActivationSparsity);

            addPpeTask(dpuElementWiseOp, opType);
            auto ppeLayerType = mv::PPELayerType(opType);
            auto ppeFixedFunction = mv::PPEFixedFunction();
            ppeFixedFunction.addLayer(ppeLayerType);
            auto ppeTask = mv::PPETask(ppeFixedFunction);
            dpuElementWiseOp->set<mv::PPETask>("PPETask", ppeTask);

            if(!splitStrategy.empty())
            {
                //NOTE:Elwise can not be SplitOverK
               dpuElementWiseOp->set<std::string>("splitStrategy", splitStrategy);
               if (splitStrategy == "HKSwitch")
                    dpuElementWiseOp->set<bool>("multiCast", true);
                else
                   dpuElementWiseOp->set<bool>("multiCast", false);
            }

            dpuElementWise->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            dpuElementWise->set<mv::DType>("dType", outputTensorType);

            mv::setOutputDataFlow(om, dpuElementWise, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(dpuElementWiseOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(dpuElementWiseOp), outputControlFlows);
        }
        //TODO: Fully connected
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

            // Correct order of strings if necessary
            auto old_order_str = input->getOrder().toString();
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

            upaPermute->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            setOutputDataFlow(om, upaPermute, outputDataFlows);
            setInputControlFlow(cm, cm.switchContext(upaPermuteOp), inputControlFlows);
            setOutputControlFlow(cm, cm.switchContext(upaPermuteOp), outputControlFlows);

        }
        else
            ++opIt;
    }
}

static void addPpeTask(mv::Data::OpListIterator &opIt, std::string ppeTaskType, double leakyAlpha)
{
    auto ppeLayerType = mv::PPELayerType(ppeTaskType);
    int8_t ppeMult;
    uint8_t ppeShift;
    auto ppeFixedFunction = mv::PPEFixedFunction();

    if (ppeTaskType == "LPRELU")
    {
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
        ppeFixedFunction = mv::PPEFixedFunction(ppeMult, ppeShift);
    }

    ppeFixedFunction.addLayer(ppeLayerType);
    auto ppeTask = mv::PPETask(ppeFixedFunction);
    opIt->set<mv::PPETask>("PPETask", ppeTask);

}

static void computeClampValues(mv::Data::OpListIterator &opIt)
{

    fixed_rep fixed_obj;
    computeClampMinimum(opIt, fixed_obj);
    computeClampMaximum(opIt, fixed_obj);
}

static void computeClampMinimum(mv::Data::OpListIterator &opIt, fixed_rep &fixed_min)
{
    if (opIt->hasAttr("minimum"))
    {
        if (opIt->getOutputTensor()[0]->getDType() == mv::DType("Float16"))
        {
            fixed_min.float_rep = opIt->get<double>("minimum");
            opIt->set<int32_t>("clampMinimum", fixed_min.int_rep);
        }
        else
            opIt->set<int32_t>("clampMinimum", opIt->get<int32_t>("minimum"));
    }
}

static void computeClampMaximum(mv::Data::OpListIterator &opIt, fixed_rep &fixed_max)
{
    if (opIt->hasAttr("maximum"))
    {
        if (opIt->getOutputTensor()[0]->getDType() == mv::DType("Float16"))
        {
            fixed_max.float_rep = opIt->get<double>("maximum");
            opIt->set<int32_t>("clampMaximum", fixed_max.int_rep);
        }
        else
            opIt->set<int32_t>("clampMaximum", opIt->get<int32_t>("maximum"));
    }
}
