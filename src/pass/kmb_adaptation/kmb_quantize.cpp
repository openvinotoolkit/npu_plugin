#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"


static void kmbQuantizeConversionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);
static void configureOutputPrecisionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(KMBQuantizeConversion)
        .setFunc(kmbQuantizeConversionFcn)
        .setDescription(
            "This pass inserts Quantize conversion layers between DPUTask-to-UPATask transitions (& vice-versa)."
        );

        MV_REGISTER_PASS(ConfigureOutputPrecision)
        .setFunc(configureOutputPrecisionFcn)
        .setDescription(
            "This pass inserts Quantize conversion layers in order to guarantee the appropriate precision."
        );
    }

}

void addQuantizationLayers(mv::OpModel om, std::vector<mv::Data::OpListIterator>& tasks, const mv::DType& dtypeNeededInInput)
{
    for(auto& task : tasks)
    {
        if (task->hasAttr("taskOp") && task->get<std::string>("taskOp") == "Quantize")
        {
            auto quantParams = task->get<mv::QuantizationParams>("quantParams");
            auto output = task->getOutputTensor(0);
            output->set<mv::QuantizationParams>("quantParams", quantParams);
            continue;
        }

        auto inputFlow = task.leftmostInput();
        auto outputDType = task->getOutputTensor(0)->getDType();
        std::size_t id = 0;
        while(inputFlow != om.flowEnd())
        {
            auto tensor = inputFlow->getTensor();
            auto tensorDType = tensor->getDType();

            // NOTE: Maybe here a check for mixed precision should be added
            if(!tensor->isPopulated() && tensorDType != dtypeNeededInInput)
            {
                //if the previous Op is "Align" need to place it after the quantize
                auto previousOpIt = om.getSourceOp(tensor);
                bool alignCase = false;
                if (previousOpIt->getOpType() == "Align")
                {
                    tensor = previousOpIt->getInputTensor()[0];
                    alignCase = true;
                }
                auto quantize = om.uPATaskQuantize({tensor}, outputDType,
                            tensor->get<mv::QuantizationParams>("quantParams"), "Quantize" + task->getName() + std::to_string(id));
                quantize->set<std::string>("splitStrategy",
                            tensor->get<std::string>("splitStrategy"));
                auto quantizeOp = om.getSourceOp(quantize);
                quantizeOp->set<unsigned>("opId", task->get<unsigned>("opId"));

                if (alignCase)
                {
                    auto backup = previousOpIt.leftmostInput();
                    auto slot = backup->get<size_t>("sinkInput");
                    ++inputFlow;
                    om.undefineFlow(backup);
                    previousOpIt->setInputTensor(quantize, slot, false);
                    previousOpIt->getOutputTensor(0)->setDType(outputDType);
                    om.defineFlow(quantize, previousOpIt, slot);
                }
                else
                {
                    auto backup = inputFlow;
                    auto slot = backup->get<size_t>("sinkInput");
                    ++inputFlow;
                    om.undefineFlow(backup);
                    task->setInputTensor(quantize, slot, false);
                    om.defineFlow(quantize, task, slot);
                }


                id++;
            }
            else
                ++inputFlow;
        }
    }
}

void addSliceQuantizationLayer(mv::OpModel om, std::vector<mv::Data::OpListIterator>& slices, const mv::DType& dtypeNeededInInput)
{
    std::vector <mv::Data::TensorIterator> sliceInputs = {};
    std::map <std::string, std::vector<mv::Data::OpListIterator>> sliceLeafs;
    std::map <std::string, std::vector<mv::Data::FlowListIterator>> sliceFlows;
    std::vector<mv::Data::OpListIterator> slicesToRemove;

    for (auto& slice : slices)
    {
        auto it = std::find (sliceInputs.begin(), sliceInputs.end(), slice->getInputTensor()[0]);
        if (it != sliceInputs.end())
        {
            sliceLeafs[slice->getInputTensor()[0]->getName()].push_back(slice);
            slicesToRemove.push_back(slice);
        }
        else
        {
            sliceInputs.push_back(slice->getInputTensor()[0]);
            auto previousOpIt = om.getSourceOp(slice->getInputTensor(0));
            for (auto sinkFlow = previousOpIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
            {
                if (sinkFlow.sink()->getOpType() == "Slice")
                    sliceFlows[slice->getInputTensor()[0]->getName()].push_back(sinkFlow);
            }
        }
    }

    for (auto slice: slicesToRemove)
    {
        slices.erase(std::remove(slices.begin(), slices.end(), slice), slices.end());
    }

    for(auto& slice : slices)
    {
        auto inputFlow = slice.leftmostInput();
        auto outputDType = slice->getOutputTensor(0)->getDType();
        std::size_t id = 0;
        while(inputFlow != om.flowEnd())
        {
            auto tensor = inputFlow->getTensor();
            auto tensorDType = tensor->getDType();

            // NOTE: Maybe here a check for mixed precision should be added
            if(!tensor->isPopulated() && tensorDType != dtypeNeededInInput)
            {
                //before adding UPATask, check if any of the other outputs of the tensor has already been quantized
                auto previousOpIt = om.getSourceOp(tensor);
                mv::Data::TensorIterator quantize;
                bool alreadyQuantized = false;
                for (auto sinkFlow = previousOpIt.leftmostOutput(); sinkFlow != om.flowEnd(); ++sinkFlow)
                {
                    auto task = sinkFlow.sink();
                    if (task->getOpType() == "UPATask" && task->hasAttr("taskOp") && task->get<std::string>("taskOp") == "Quantize")
                    {
                        quantize = task->getOutputTensor()[0];
                        alreadyQuantized = true;
                        break;
                    }

                }

                if (!alreadyQuantized)
                {
                    quantize = om.uPATaskQuantize({tensor}, outputDType,
                            tensor->get<mv::QuantizationParams>("quantParams"), "Quantize" + slice->getName() + std::to_string(id));
                    quantize->set<std::string>("splitStrategy",
                            tensor->get<std::string>("splitStrategy"));
                    auto quantizeOp = om.getSourceOp(quantize);
                    quantizeOp->set<unsigned>("opId", slice->get<unsigned>("opId"));
                }
                auto backup = inputFlow;
                auto slot = backup->get<size_t>("sinkInput");
                ++inputFlow;
                for (auto flow:sliceFlows[tensor->getName()])
                {
                    om.undefineFlow(flow);
                }
                slice->setInputTensor(quantize, slot, false);
                om.defineFlow(quantize, slice, slot);
                for (auto op:sliceLeafs[tensor->getName()])
                {
                    op->setInputTensor(quantize, slot, false);
                    om.defineFlow(quantize, op, slot);
                }
                id++;
            }
            else
                ++inputFlow;
        }
    }
}

static void kmbQuantizeConversionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);

    auto dpuTasks = om.getOps("DPUTask");
    std::vector<mv::Data::OpListIterator> dpuTasksFP16 = {};
    for (auto& dpuTask : dpuTasks)
    {
        if (dpuTask->hasAttr("softwareExecuted") && dpuTask->get<bool>("softwareExecuted"))
        {
            dpuTasksFP16.push_back(dpuTask);
            dpuTasks.erase(std::remove(dpuTasks.begin(), dpuTasks.end(), dpuTask), dpuTasks.end());
        }
    }

    auto upaTasks = om.getOps("UPATask");

    // NOTE: At this moment in the model, all the concats are implicit
    auto implicitConcats = om.getOps("ImplicitConcat");
    // NOTE: For now only operations with U8/DPU Tasks are streamed
    auto slices = om.getOps("Slice");

    auto U8 = mv::DType("UInt8");
    auto FP16 = mv::DType("Float16");

    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    addQuantizationLayers(om, upaTasks, FP16);

    bool DPUTasksinSW = globalParams->hasAttr("DPUTasksinFloat") ? globalParams->get<bool>("DPUTasksinFloat") : false;
    if (!DPUTasksinSW)
    {
        addQuantizationLayers(om, dpuTasksFP16, FP16);
        addQuantizationLayers(om, dpuTasks, U8);
        addSliceQuantizationLayer(om, slices, U8);
    }
    // NOTE: Concat have the extra requirement that output tensor and input tensor have to match their DType, so
    // we split them in two vectors

    std::vector<mv::Data::OpListIterator> implicitConcatsU8;
    std::vector<mv::Data::OpListIterator> implicitConcatsFP16;

    for(auto& implicitConcat: implicitConcats)
    {
        auto outputDType = implicitConcat->getOutputTensor(0)->getDType();
        if(outputDType == U8)
            implicitConcatsU8.push_back(implicitConcat);
        else if(outputDType == FP16)
            implicitConcatsFP16.push_back(implicitConcat);
    }

    addQuantizationLayers(om, implicitConcatsU8, U8);
    addQuantizationLayers(om, implicitConcatsFP16, FP16);
}

static void configureOutputPrecisionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    //Note: The idea is that this pass is used mainly for validation when different precision is needed, so no problem
    //to do always the type conversion with quantize
    mv::OpModel om(model);
    //Note: Always a vector of one element
    auto outputOp = om.getOps("Output");
    if (outputOp[0]->hasAttr("precision") && outputOp[0]->get<mv::DType>("precision") != mv::DType("Default"))
    {
        auto inputTypeofOutput = outputOp[0]->getInputTensor(0)->getDType();
        auto wantedPrecision = outputOp[0]->get<mv::DType>("precision");
        if (inputTypeofOutput != wantedPrecision)
        {
            if (om.getSourceOp(outputOp[0]->getInputTensor(0))->isImplicit())
            {
                for (std::size_t i = 0; i < om.getSourceOp(outputOp[0]->getInputTensor(0))->getInputTensor().size(); i++)
                    om.getSourceOp(outputOp[0]->getInputTensor(0))->getInputTensor(i)->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::DDR);
            }
            auto quantize = om.uPATaskQuantize({outputOp[0]->getInputTensor(0)}, wantedPrecision,
                        outputOp[0]->getInputTensor(0)->get<mv::QuantizationParams>("quantParams"), "Precision" + outputOp[0]->getName());
            quantize->set<std::string>("splitStrategy",
                        outputOp[0]->getInputTensor(0)->get<std::string>("splitStrategy"));
            quantize->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::OUTPUT);
            outputOp[0]->getInputTensor(0)->set<mv::Tensor::MemoryLocation>("Location",mv::Tensor::MemoryLocation::DDR);
            outputOp[0]->getInputTensor(0)->set<mv::QuantizationParams>("quantParams",
                                outputOp[0]->getInputTensor(0)->get<mv::QuantizationParams>("quantParams"));
            auto quantizeOp = om.getSourceOp(quantize);
            quantizeOp->set<unsigned>("opId", outputOp[0]->get<unsigned>("opId") - 1);
            om.undefineFlow(outputOp[0].leftmostInput());
            outputOp[0]->setInputTensor(quantize, 0, false);
            om.defineFlow(quantize, outputOp[0], 0);
        }
    }
    else
        return;
}
