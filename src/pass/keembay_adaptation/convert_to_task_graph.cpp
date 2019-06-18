#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/target/keembay/ppe_task.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/pass/pass_utils.hpp"

static const std::array<unsigned short, 2> FAKE_KERNEL = {1,1};
static const std::array<unsigned short, 2> FAKE_STRIDE = {1,1};

static void convertOpsToTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(ConvertOpsToTasks)
            .setFunc(convertOpsToTasksFcn)
            .setDescription(
                "Replace all convolution operations with DPU tasks.\n"
                "Assume each convolution can be done with DPU on KMB.\n"
                "Assume each convolution should be done on DPU.");
    }
}

void convertOpsToTasksFcn(const mv::pass::PassEntry& , mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&)
{
    mv::OpModel om(model);

    auto addFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const std::string& s){ return om.dPUTaskAdd(vec,quantParams,s);};
    auto subFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const std::string& s){ return om.dPUTaskSubtract(vec,quantParams,s);};
    auto multFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const mv::QuantizationParams& quantParams, const std::string& s){ return om.dPUTaskMultiply(vec,quantParams,s);};

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
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

            auto input = opIt->getInputTensor(0);
            auto kernel = opIt->getInputTensor(1);
            auto opId = opIt->get<unsigned>("opId");

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");

            auto name = opIt->getName();
            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");

            std::string biasName, splitStrategy, workloadStrategyMPEMode;
            int workloadStrategyNWorkloads = -1;

            unsigned group = 1;
            if (opType == "Conv")
                group = opIt->get<unsigned>("group");

            if (opIt->hasAttr("bias"))
                biasName = opIt->get<std::string>("bias");

            if(opIt->hasAttr("splitStrategy"))
                splitStrategy = opIt->get<std::string>("splitStrategy");

            if (opIt->hasAttr("WorkloadStrategy_nWorkloads"))
                workloadStrategyMPEMode = opIt->get<std::string>("WorkloadStrategy_MPE_mode");

            if (opIt->hasAttr("WorkloadStrategy_nWorkloads"))
                workloadStrategyNWorkloads = opIt->get<int>("WorkloadStrategy_nWorkloads");

            std::array<unsigned short, 2> kernelSize = {kernel->getShape()[mv::KERNEL_WIDTH], kernel->getShape()[mv::KERNEL_HEIGHT]};

            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            mv::Data::TensorIterator dpuConv;
            if(opType == "Conv")
                dpuConv = om.dPUTaskConv({input, kernel}, strides, padding, dilationFactor, group, quantParams, mv::createDPUTaskName(name));
            else
                dpuConv = om.dPUTaskDepthwiseConv({input, kernel}, strides, padding, dilationFactor, quantParams, mv::createDPUTaskName(name));

            auto dpuConvOp = om.getSourceOp(dpuConv);
            dpuConvOp->set<unsigned>("opId", opId);
            dpuConvOp->set<bool>("hasWeights", true);
            dpuConvOp->set<std::array<unsigned short, 2>>("kSize", kernelSize);


            if(!biasName.empty())
               dpuConvOp->set<std::string>("bias", biasName);
            if(!splitStrategy.empty())
               dpuConvOp->set<std::string>("splitStrategy", splitStrategy);
            if(!workloadStrategyMPEMode.empty())
                dpuConvOp->set<std::string>("WorkloadStrategy_MPE_mode", workloadStrategyMPEMode);
            if(workloadStrategyNWorkloads != -1)
                dpuConvOp->set<int>("WorkloadStrategy_nWorkloads", workloadStrategyNWorkloads);

            if(opType == "Conv")
            {
                if(kernel->getShape()[mv::KERNEL_INPUT_CHANNELS] < 16)
                {
                    dpuConvOp->erase("taskOp");
                    dpuConvOp->set<std::string>("taskOp", "ChannelMajorConvolution");
                }
            }

            //dpuConv->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            setOutputDataFlow(om, dpuConv, outputDataFlows);
        }
        else if (opType == "MaxPool")
        {
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

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

            std::string splitStrategy;

            if(opIt->hasAttr("splitStrategy"))
                splitStrategy = opIt->get<std::string>("splitStrategy");

            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            auto dpuPool = om.dPUTaskMaxPool({input}, kernelSize, strides, padding,
                               exclude_pad, auto_pad, rounding_type, quantParams, mv::createDPUTaskName(name));
            auto dpuPoolOp = om.getSourceOp(dpuPool);
            dpuPoolOp->set<unsigned>("opId", opId);
            dpuPoolOp->set<bool>("hasWeights", false);

            if(!splitStrategy.empty())
               dpuPoolOp->set<std::string>("splitStrategy", splitStrategy);
            //dpuPool->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            setOutputDataFlow(om, dpuPool, outputDataFlows);
        }
        else if (opType == "Add" || opType == "Subtract" || opType == "Multiply")
        {
            auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");

            auto input1 = opIt->getInputTensor(0);
            auto input2 = opIt->getInputTensor(1);
            std::vector<mv::Data::TensorIterator> inputs;
            inputs.push_back(input1);
            inputs.push_back(input2);
            auto name = opIt->getName();

            auto quantParams = opIt->get<mv::QuantizationParams>("quantParams");

            auto opId = opIt->get<unsigned>("opId");

            std::string splitStrategy;

            if(opIt->hasAttr("splitStrategy"))
                splitStrategy = opIt->get<std::string>("splitStrategy");

            auto outputDataFlows = mv::getOutputDataFlow(om, opIt);

            auto dpuElementWiseFunctor = (dpuTaskMap.at(opType));
            auto dpuElementWise = dpuElementWiseFunctor(inputs, quantParams, mv::createDPUTaskName(name));
            auto dpuElementWiseOp = om.getSourceOp(dpuElementWise);
            dpuElementWiseOp->set<unsigned>("opId", opId);
            dpuElementWiseOp->set<bool>("hasWeights", false);
            dpuElementWiseOp->set<std::array<unsigned short, 2>>("kSize", FAKE_KERNEL);
            dpuElementWiseOp->set<std::array<unsigned short, 2>>("stride", FAKE_STRIDE);

            auto ppeLayerType = mv::PPELayerType(opType);
            auto ppeFixedFunction = mv::PPEFixedFunction();
            ppeFixedFunction.addLayer(ppeLayerType);
            auto ppeTask = mv::PPETask(ppeFixedFunction);
            dpuElementWiseOp->set<mv::PPETask>("PPETask", ppeTask);

            if(!splitStrategy.empty())
               dpuElementWiseOp->set<std::string>("splitStrategy", splitStrategy);

            //dpuElementWiseOp->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
            mv::setOutputDataFlow(om, dpuElementWise, outputDataFlows);
        }
        //TODO: Fully connected
        else
            ++opIt;
    }
}


