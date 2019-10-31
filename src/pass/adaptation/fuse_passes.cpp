#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include <functional>

static void fuseBiasFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel& model, std::string opType);
static void fuseUsualPPEFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType);
static void fuseMinimumFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, std::string opType);
static void fuseMaximumFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, std::string opType);
static void fuseScaleFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, std::string opType);
static void fuseBatchNormFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, std::string opType);
static void fusePostOpsFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);


namespace mv
{
    namespace pass
    {

        MV_REGISTER_PASS(FusePostOps)
        .setFunc(fusePostOpsFcn)
        .setDescription(
            "Fuses all the ops that will be converted to PPE Tasks and can be handled through hardware. "
            "Scale, Batch Norm from My-X\n"
        );
    }
}

void fusePostOpsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    using namespace mv;

    OpModel om(model);
    DataModel dm(model);

    std::vector<std::string> fuse_types = {"Bias", "Sigmoid", "Relu", "LeakyRelu", "Power", "MinimumDouble",
                                           "MinimumInt", "MaximumDouble", "MaximumInt"};
    std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operationsOfType = om.getOpsOfTypes(fuse_types);
    std::size_t totalPPETasks = 0;
    for (auto it = operationsOfType.begin(); it != operationsOfType.end(); it++)
    {
        totalPPETasks += it->second.size();
    }

    UNUSED(fuseScaleFcn);
    UNUSED(fuseBatchNormFcn);
    auto fuseBias = [](mv::Data::OpListIterator &opIt, mv::ComputationModel& cm, std::string &empty){ return fuseBiasFcn(opIt, cm, empty);};
    auto fuseSigmoid = [](mv::Data::OpListIterator &opIt, mv::ComputationModel& cm, std::string &empty){ return fuseUsualPPEFcn(opIt, cm, empty);};
    auto fuseRelu = [](mv::Data::OpListIterator &opIt, mv::ComputationModel& cm, std::string &empty){ return fuseUsualPPEFcn(opIt, cm, empty);};
    auto fuseLeakyRelu = [](mv::Data::OpListIterator &opIt, mv::ComputationModel& cm, std::string &empty){ return fuseUsualPPEFcn(opIt, cm, empty);};
    auto fusePower = [](mv::Data::OpListIterator &opIt, mv::ComputationModel& cm, std::string &empty){ return fuseUsualPPEFcn(opIt, cm, empty);};
    auto fuseMinimum = [](mv::Data::OpListIterator &opIt, mv::ComputationModel& cm, std::string &empty){ return fuseMinimumFcn(opIt, cm, empty);};
    auto fuseMaximum = [](mv::Data::OpListIterator &opIt, mv::ComputationModel& cm, std::string &empty){ return fuseMaximumFcn(opIt, cm, empty);};

    std::unordered_map<std::string, std::function<void(mv::Data::OpListIterator &, mv::ComputationModel& , std::string &)>> fuseTaskMap =
                                       {{"Bias", fuseBias},
                                        {"Sigmoid", fuseSigmoid},
                                        {"Relu", fuseRelu},
                                        {"LeakyRelu", fuseLeakyRelu},
                                        {"Power", fusePower},
                                        {"MinimumDouble", fuseMinimum},
                                        {"MinimumInt", fuseMinimum},
                                        {"MaximumDouble", fuseMaximum},
                                        {"MaximumInt", fuseMaximum}};

    auto opIt = om.getInput();
    while (totalPPETasks > 0)
    {
        auto opType = opIt->getOpType();
        if (fuseTaskMap.find(opType) != fuseTaskMap.end())
        {
            auto fuseFunctor = (fuseTaskMap.at(opType));
            fuseFunctor(opIt, model, opType);
            totalPPETasks--;
        }
        ++opIt;
    }

}

mv::Data::OpListIterator linkNewOperationsFuse(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel om, mv::Data::OpListIterator opIt)
{
    //Important: do not change the order of this ops
    std::vector<mv::Data::OpListIterator> opsToLink;
    std::vector<std::size_t> inputSlots;
    for (mv::Data::FlowSiblingIterator sinkFlow(opIt.leftmostOutput()); sinkFlow != om.flowEnd(); ++sinkFlow)
    {
        opsToLink.push_back(sinkFlow.sink());
        inputSlots.push_back(sinkFlow->get<std::size_t>("sinkInput"));
    }

    auto paramOp = opIt.leftmostParent();
    while(paramOp != om.opEnd())
    {
        if (paramOp->getOpType() == "Constant" || paramOp->getOpType() == "ConstantInt" || paramOp->getOpType() == "ConstantDataElement")
        {
            auto backUp = paramOp;
            ++paramOp;
            om.removeOp(backUp);
        }
        else
            ++paramOp;
    }

    om.removeOp(opIt);
    opIt = parentOpIt;

    for (unsigned j = 0; j < opsToLink.size(); ++j)
    {
        opsToLink[j]->setInputTensor(sourceTensor, inputSlots[j], false);
        om.defineFlow(sourceTensor, opsToLink[j], inputSlots[j]);
    }

    return opIt;
}

void fuseBiasFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    if (parentOpIt->getOpType() == "Conv" ||
        parentOpIt->getOpType() == "FullyConnected" ||
        parentOpIt->getOpType() == "DepthwiseConv")
    {
        auto bias = *opIt->getInputTensor(1);
        auto biasOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
        if (parentOpIt->hasAttr("bias"))
        {
            auto biasTensor = model.getTensor(parentOpIt->get<std::string>("bias"));
            biasTensor->add(bias);
        }
        else
        {
            std::string biasTensorName = mv::createBiasName(parentOpIt->getName());
            mv::Data::TensorIterator biasTensor;
            if (bias.hasAttr("quantParams"))
                biasTensor = dm.defineTensor(mv::Tensor(biasTensorName, bias.getShape(), bias.getDType(), bias.getOrder(), bias.getData(), bias.get<mv::QuantizationParams>("quantParams")) );
            else
                biasTensor = dm.defineTensor(mv::Tensor(biasTensorName, bias.getShape(), bias.getDType(), bias.getOrder(), bias.getData()) );
            om.addAttr(parentOpIt, "bias", biasTensor->getName());
        }
        auto sourceTensor = parentOpIt->getOutputTensor(0);
        opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
        if (biasOutputMemoryLocation.isForced())
        {
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", biasOutputMemoryLocation);
        }
    }
}

void fuseScaleFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    auto scaleOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    if (parentOpIt->getOpType() == "Conv")
    {
        auto scale = *opIt->getInputTensor(1);
        parentOpIt->getInputTensor(1)->multiply(scale);
        if (parentOpIt->hasAttr("bias"))
        {
            auto biasTensor = model.getTensor(parentOpIt->get<std::string>("bias"));
            biasTensor->multiply(scale);
        }
        auto sourceTensor = parentOpIt->getOutputTensor(0);
        opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
        if (scaleOutputMemoryLocation.isForced())
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", scaleOutputMemoryLocation);
    }
}

void fuseUsualPPEFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    mv::OpModel om(model);
    auto ppeOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    parentOpIt->set<std::string>("postOpType", opType);
    if (opType == "LeakyRelu")
        parentOpIt->set<double>("alpha", opIt->get<double>("alpha"));

    auto sourceTensor = parentOpIt->getOutputTensor(0);
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (ppeOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", ppeOutputMemoryLocation);
}

void fuseMinimumFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);
    auto minimumOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    parentOpIt->set<std::vector<std::string>>("postOpTypes", {"Minimum"});
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    if (sourceTensor->getDType() == mv::DType("Float16"))
        parentOpIt->set<double>("minimum", opIt->get<double>("minimum"));
    else
        parentOpIt->set<int64_t>("minimum", opIt->get<int64_t>("minimum"));

    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (minimumOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", minimumOutputMemoryLocation);
}

void fuseMaximumFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);
    auto maximumOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    std::vector<std::string> postOpTypes = {};
    if (parentOpIt->hasAttr("postOpTypes"))
        postOpTypes = parentOpIt->get<std::vector<std::string>>("postOpTypes");

    postOpTypes.push_back("Maximum");
    parentOpIt->set<std::vector<std::string>>("postOpTypes", postOpTypes);
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    if (sourceTensor->getDType() == mv::DType("Float16"))
        parentOpIt->set<double>("maximum", opIt->get<double>("maximum"));
    else
        parentOpIt->set<int64_t>("maximum", opIt->get<int64_t>("maximum"));

    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (maximumOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", maximumOutputMemoryLocation);
}

void fuseBatchNormFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, std::string opType)
{
    UNUSED(opType);
    mv::OpModel om(model);
    auto outputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto batchNormName = opIt->getName();
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    auto bnMean = *opIt->getInputTensor(1);
    auto bnVar = *opIt->getInputTensor(2);
    auto bnOffset = *opIt->getInputTensor(3);
    auto bnScale = *opIt->getInputTensor(4);
    double bnEps = opIt->get<double>("eps");
    auto scaleParam = mv::math::divide(bnScale, mv::math::sqrt(mv::math::add(bnVar, bnEps)));
    auto offsetParam = mv::math::subtract(bnOffset, mv::math::multiply(bnMean, scaleParam));
    auto offset = om.constantDataElement(offsetParam.getData(), offsetParam.getShape(), offsetParam.getDType(),
        offsetParam.getOrder(),{{},{},{},{}}, batchNormName + "_offset");

    mv::Data::TensorIterator sourceTensor;

    if (bnMean.getShape().ndims() == 1)
    {
        if (parentOpIt->getOpType() == "Conv")
        {
            parentOpIt->getInputTensor(1)->multiply(scaleParam);
            sourceTensor = parentOpIt->getOutputTensor(0);
        }
        else
        {
            auto scale = om.constantDataElement(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
            sourceTensor = om.scale(opIt->getInputTensor(0), scale);
            parentOpIt = om.getSourceOp(sourceTensor);
        }
    }
    else
    {
        auto scale = om.constantDataElement(scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
        sourceTensor = om.multiply({opIt->getInputTensor(0), scale});
        parentOpIt = om.getSourceOp(sourceTensor);
    }

    if (offsetParam.getShape().ndims() == 1)
        sourceTensor = om.bias(sourceTensor, offset);
    else
        sourceTensor = om.add({sourceTensor, offset});
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (outputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
}
