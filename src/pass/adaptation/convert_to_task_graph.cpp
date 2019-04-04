#include "include/mcm/pass/pass_registry.hpp"
#include "meta/include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static void convertOpsToTasksFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::json::Object&);
void adaptOutputDataFlow(mv::OpModel& om, mv::Data::OpListIterator& opIt, mv::Data::TensorIterator& dpuTask);

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
    mv::DataModel dm(model);

    mv::ControlModel cm(model);

    auto addFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const std::string& s){ return om.dPUTaskAdd(vec,s);};
    auto subFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const std::string& s){ return om.dPUTaskSubtract(vec,s);};
    auto multFcn = [&om](std::vector< mv::Data::TensorIterator >& vec, const std::string& s){ return om.dPUTaskMultiply(vec,s);};

    auto dpuTaskMap = std::map<std::string, std::function<mv::Data::TensorIterator (std::vector< mv::Data::TensorIterator >&, const std::string&)>>
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
            auto input = opIt->getInputTensor(0);
            auto kernel = opIt->getInputTensor(1);
            auto opId = opIt->get<unsigned>("opId");

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto dilationFactor = opIt->get<unsigned>("dilationFactor");
            auto name = opIt->getName();

            unsigned group=1;
            if (opType == "Conv")
                group = opIt->get<unsigned>("group");

            mv::Data::TensorIterator dpuConv;
            if(opType == "Conv")
                dpuConv = om.dPUTaskConv({input, kernel}, strides, padding, dilationFactor, group, "DPU_" + name);
            else
                dpuConv = om.dPUTaskDepthwiseConv({input, kernel}, strides, padding, dilationFactor, "DPU_" + name);

            auto dpuConvOp = om.getSourceOp(dpuConv);
            dpuConvOp->set<unsigned>("opId", opId);

            if (opIt->hasAttr("bias"))
            {
                auto biasTensor = dm.getTensor(opIt->get<std::string>("bias"));
                auto name = biasTensor->getName();
                om.addAttr(dpuConvOp, "bias", name);
            }

            if(opType == "Conv")
            {
                if(kernel->getShape()[2] < 16)
                {
                    dpuConvOp->erase("taskOp");
                    dpuConvOp->set<std::string>("taskOp", "ChannelMajorConvolution");
                }
            }

            adaptOutputDataFlow(om, opIt, dpuConv);
        }
        else if (opType == "MaxPool")
        {
            auto input = opIt->getInputTensor(0);
            auto opId = opIt->get<unsigned>("opId");

            auto strides = opIt->get<std::array<unsigned short, 2>>("stride");
            auto padding = opIt->get<std::array<unsigned short, 4>>("padding");
            auto kernelSize = opIt->get<std::array<unsigned short, 2>>("kSize");
            auto exclude_pad = opIt->get<bool>("exclude_pad");
            auto auto_pad = opIt->get<std::string>("auto_pad");
            auto rounding_type = opIt->get<std::string>("rounding_type");
            auto name = opIt->getName();

            auto dpuPool = om.dPUTaskMaxPool({input}, kernelSize, strides, padding,
                               exclude_pad, auto_pad, rounding_type, "DPU_" + name);
            auto dpuPoolOp = om.getSourceOp(dpuPool);
            dpuPoolOp->set<unsigned>("opId", opId);

            adaptOutputDataFlow(om, opIt, dpuPool);
        }
        else if (opType == "Add" || opType == "Subtract" || opType == "Multiply")
        {
            auto input1 = opIt->getInputTensor(0);
            auto input2 = opIt->getInputTensor(1);
            std::vector<mv::Data::TensorIterator> inputs;
            inputs.push_back(input1);
            inputs.push_back(input2);
            auto name = opIt->getName();

            auto opId = opIt->get<unsigned>("opId");

            auto dpuElementWiseFunctor = (dpuTaskMap.at(opType));
            auto dpuElementWise = dpuElementWiseFunctor(inputs, "DPU_"+name);
            auto dpuElementWiseOp = om.getSourceOp(dpuElementWise);
            dpuElementWiseOp->set<unsigned>("opId", opId);

            adaptOutputDataFlow(om, opIt, dpuElementWise);
        }
        else
            ++opIt;
    }
}

void adaptOutputDataFlow(mv::OpModel& om, mv::Data::OpListIterator &opIt, mv::Data::TensorIterator &dpuTask)
{
    for(auto output = opIt.leftmostOutput(); output != om.flowEnd(); ++output)
    {
        auto consumer = output.sink();
        auto slot = output->get<size_t>("sinkInput");
        consumer->setInputTensor(dpuTask, slot, false);
        om.defineFlow(dpuTask, consumer, slot);
    }

    auto backup = opIt;
    ++opIt;
    om.removeOp(backup);
}
