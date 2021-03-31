#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/tensor/math.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/utils/warning_manager.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <functional>

void fuseBiasFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel& model, const std::string& opType);
void fuseUsualPPEFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, const std::string& opType);
void fuseMinimumFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, const std::string& opType);
void fuseMaximumFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, const std::string& opType);
void fuseEltwiseFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, const std::string& opType);
void fuseScaleFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, const std::string& opType);
void fuseBatchNormFcn(mv::Data::OpListIterator &opIt,  mv::ComputationModel& model, const std::string& opType);
static void fusePostOpsFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

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
    mv::OpModel om(model);
    std::shared_ptr<mv::Element> globalParams = model.getGlobalConfigParams();
    std::unordered_map<std::string, std::function<void(mv::Data::OpListIterator &, mv::ComputationModel& , const std::string &)>> fuseTaskMap =
                                    {{"Bias", fuseBiasFcn},
                                    {"Sigmoid", fuseUsualPPEFcn},
                                    {"Tanh", fuseUsualPPEFcn},
                                    {"Relu", fuseUsualPPEFcn},
                                    {"Prelu", fuseUsualPPEFcn},
                                    {"LeakyRelu", fuseUsualPPEFcn},
                                    {"Mish", fuseUsualPPEFcn},
                                    {"Minimum", fuseMinimumFcn},
                                    {"Maximum", fuseMaximumFcn}};

    if (checkPPEAccuracy(model))
    {
        std::vector<mv::Data::OpListIterator> preluOperations = om.getOps("Prelu");
        if (preluOperations.size() > 0)
        {
            throw mv::OpError(preluOperations[0]->getLogID(), "no support of PRelu wiht PPEAccuracy");
        }

        std::vector<mv::Data::OpListIterator> biasOperations = om.getOps("Bias");

        for (auto bias : biasOperations)
            fuseBiasFcn(bias, model, "Bias");

        provideAccuracyinPPEs(model);
        std::vector<std::string> fuse_types = {"Sigmoid", "Tanh", "Relu", "Minimum", "Maximum"};
        std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operationsOfType = om.getOpsOfTypes(fuse_types);

        //NOTE: Iterate the fuse_types vector for correct order reason according to map
        for (auto type = fuse_types.begin(); type != fuse_types.end(); type++)
        {
            auto fuseFunctor = (fuseTaskMap.at(*type));
            for (auto opIt = operationsOfType[*type].begin(); opIt != operationsOfType[*type].end();++opIt)
                fuseFunctor(*opIt, model, *type);
        }
    }
    else
    {
        std::vector<std::string> fuse_types = {"Bias", "Sigmoid", "Tanh", "Relu", "LeakyRelu", "Mish", "Minimum", "Maximum", "Prelu"};
        std::unordered_map<std::string, std::vector<mv::Data::OpListIterator>> operationsOfType = om.getOpsOfTypes(fuse_types);

        //NOTE: Iterate the fuse_types vector for correct order reason according to map
        for (auto type = fuse_types.begin(); type != fuse_types.end(); type++)
        {
            auto fuseFunctor = (fuseTaskMap.at(*type));
            for (auto opIt = operationsOfType[*type].begin(); opIt != operationsOfType[*type].end();++opIt)
                fuseFunctor(*opIt, model, *type);
        }
    }
}

mv::Data::OpListIterator linkNewOperationsFuse(mv::Data::OpListIterator parentOpIt, mv::Data::TensorIterator sourceTensor, mv::OpModel & om, mv::Data::OpListIterator opIt)
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

void fuseBiasFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, const std::string& /*opType*/)
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));
    if (parentOpIt->getOpType() == "Conv" ||
        parentOpIt->getOpType() == "FullyConnected" ||
        parentOpIt->getOpType() == "DepthwiseConv" ||
        parentOpIt->getOpType() == "Deconv" ||
        parentOpIt->getOpType() == "MaxPool")
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
        sourceTensor->setDType(opIt->getOutputTensor(0)->getDType());
        sourceTensor->setQuantParams(opIt->getOutputTensor(0)->getQuantParams());
        opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
        if (biasOutputMemoryLocation.isForced())
        {
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", biasOutputMemoryLocation);
        }
    }
}

void fuseScaleFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, const std::string& /*opType*/)
{
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
        sourceTensor->setDType(opIt->getOutputTensor(0)->getDType());
        sourceTensor->setQuantParams(opIt->getOutputTensor(0)->getQuantParams());
        opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
        if (scaleOutputMemoryLocation.isForced())
            opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", scaleOutputMemoryLocation);
    }
}

/**
 * @brief fuse PPE function to parent op
 */
void fuseUsualPPEFcn( mv::Data::OpListIterator& opIt, mv::ComputationModel& model, const std::string& opType )
{
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto ppeOutputMemoryLocation = opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOp = om.getSourceOp(opIt->getInputTensor(mv::IO_TENSOR_INPUT));

    std::vector<mv::Data::OpListIterator> fusableParents;
    auto isActivationAgnostic = [](mv::Data::OpListIterator &op)
        {return op->isImplicit() || op->getOpType() == "Concat";};

    if (!isActivationAgnostic(parentOp))
        fusableParents.push_back(parentOp);
    else {
        std::queue<mv::Data::OpListIterator> op_itr_bfs;
        op_itr_bfs.push(parentOp);
        // BFS the implicit ops subtree to find all fusable parents
        while (!op_itr_bfs.empty()) {
            auto current_op_itr = op_itr_bfs.front();
            for(auto parentIt = current_op_itr.leftmostParent();
                parentIt != om.opEnd(); ++parentIt) {
                if (parentIt->isImplicit() || parentIt->getOpType() == "Concat") {
                    op_itr_bfs.push(parentIt);
                } else {
                    fusableParents.push_back(parentIt);
                }
            }
            op_itr_bfs.pop();
        }
    }

    // Check number of children of each fusable parent;
    // normally if all children are the same opType,
    // one could proceed to attempt fuse all siblings.
    // Have this as a future optimization, for now just mark it as
    // software executed.
    if (std::find_if(fusableParents.begin(), fusableParents.end(),
            [] (mv::Data::OpListIterator &op) {return op.childrenSize() > 1 || !op->isHardwarizable();})
        != fusableParents.end())
    {
        opIt->set<bool>("softwareExecuted", true);
        return;
    }

    // Proceed with fusign postOp into each parentOp
    for(auto parentIt : fusableParents)
    {
        std::vector<std::string> postOpTypes;
        if (parentIt->hasAttr("postOpTypes"))
            postOpTypes = parentIt->get<std::vector<std::string>>("postOpTypes");
        postOpTypes.push_back(opType);
        parentIt->set<std::vector<std::string>>("postOpTypes", postOpTypes);

        if (opType == "LeakyRelu")
        {
            parentIt->set<double>("leakyAlpha", opIt->get<double>("alpha"));
        }
        else if (opType == "Prelu")
        {
            // slope is the second input of PRelu
            auto parentOpIt1 = om.getSourceOp(opIt->getInputTensor(1));
            const std::string& parentOpType = parentOpIt1->getOpType();

            // Constant Operator is only support
            if ((parentOpType == "Constant") || (parentOpType == "ConstantInt") || (parentOpType == "ConstantDataElement"))
            {
                // TODO
                // check how to handle non constant slopes as constant slops are only supported in the current implementation
                throw mv::OpError(parentOpIt1->getLogID(), "Non Const slopes of PReLU are not supported");
            }

            // output densor of the const output tensor
            auto slopeSourceTensor = parentOpIt1->getOutputTensor(mv::IO_TENSOR_OUTPUT);

            std::vector<mv::DataElement> data = slopeSourceTensor->getData();
        
            std::vector<double> slopes(data.size());
            if (slopeSourceTensor->getDType() != mv::DType("Float16"))
            {
                // slopes are converted into fp16 before fusing passing
                throw mv::OpError(parentOpIt1->getLogID(), "float16 data type is expected");
            }

            // converting slopes into double to use them for quantization param calculation later
            std::transform(data.begin(), data.end(), slopes.begin(), 
                [](const int64_t& arg){ return static_cast<double>(mv::fp16_to_fp32(static_cast<uint16_t>(arg))); });

            // for quantization
            parentIt->set<std::vector<double>>( "slopes", slopes );

            // Note
            // Constant is removed in linkNewOperationsFuse
        }   
    }

    // Link direct postOp parent with postOp consumers
    auto sourceTensor = parentOp->getOutputTensor(mv::IO_TENSOR_OUTPUT);
    sourceTensor->setDType(opIt->getOutputTensor(0)->getDType());
    sourceTensor->setQuantParams(opIt->getOutputTensor(0)->getQuantParams());
    opIt = linkNewOperationsFuse(parentOp, sourceTensor, om, opIt);
    if (ppeOutputMemoryLocation.isForced())
        opIt->getOutputTensor(mv::IO_TENSOR_OUTPUT)->set<mv::Tensor::MemoryLocation>("Location", ppeOutputMemoryLocation);
}

void fuseEltwiseFcn(mv::Data::OpListIterator &opIt1, mv::ComputationModel &model, const std::string& /*opType*/)
{
    std::unordered_map<std::string, std::function<void(mv::Data::OpListIterator &, mv::ComputationModel& , std::string &)>> fuseEltwiseMap =
                                       {{"Minimum", fuseMinimumFcn},
                                        {"Maximum", fuseMaximumFcn},
                                        {"Power", fuseUsualPPEFcn}};

    auto eltwiseType = opIt1->get<std::string>("eltwiseType");
    auto functor = fuseEltwiseMap.find(eltwiseType);
    if(functor != fuseEltwiseMap.end())
        functor->second(opIt1, model, eltwiseType);
}

void fuseMinimumFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, const std::string& /*opType*/)
{
    mv::OpModel om(model);

    auto minimumOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

    double minimumValue = opIt->get<double>("minimum");
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    parentOpIt->set<double>("Minimum", minimumValue);

    sourceTensor->setDType(opIt->getOutputTensor(0)->getDType());
    sourceTensor->setQuantParams(opIt->getOutputTensor(0)->getQuantParams());
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (minimumOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", minimumOutputMemoryLocation);
}

void fuseMaximumFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, const std::string& /*opType*/)
{
    mv::OpModel om(model);

    auto maximumOutputMemoryLocation = opIt->getOutputTensor(0)->get<mv::Tensor::MemoryLocation>("Location");
    auto parentOpIt = om.getSourceOp(opIt->getInputTensor(0));

    double maximumValue = opIt->get<double>("maximum");
    auto sourceTensor = parentOpIt->getOutputTensor(0);
    parentOpIt->set<double>("Maximum", maximumValue);

    sourceTensor->setDType(opIt->getOutputTensor(0)->getDType());
    sourceTensor->setQuantParams(opIt->getOutputTensor(0)->getQuantParams());
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (maximumOutputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", maximumOutputMemoryLocation);
}

void fuseBatchNormFcn(mv::Data::OpListIterator &opIt, mv::ComputationModel &model, const std::string& /*opType*/)
{
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
    auto offset = om.constantDataElement(batchNormName + "_offset",
                        offsetParam.getData(), offsetParam.getShape(),
                        offsetParam.getDType(), offsetParam.getOrder());

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
            auto scale = om.constantDataElement("", scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
            sourceTensor = om.scale("", opIt->getInputTensor(0), scale);
            parentOpIt = om.getSourceOp(sourceTensor);
        }
    }
    else
    {
        auto scale = om.constantDataElement("", scaleParam.getData(), scaleParam.getShape(), scaleParam.getDType(), scaleParam.getOrder());
        sourceTensor = om.eltwise("", {opIt->getInputTensor(0), scale}, "Multiply");
        parentOpIt = om.getSourceOp(sourceTensor);
    }

    if (offsetParam.getShape().ndims() == 1)
        sourceTensor = om.bias("", sourceTensor, offset);
    else
        sourceTensor = om.eltwise("", {sourceTensor, offset}, "Add");
    sourceTensor->setDType(opIt->getOutputTensor(0)->getDType());
    sourceTensor->setQuantParams(opIt->getOutputTensor(0)->getQuantParams());
    opIt = linkNewOperationsFuse(parentOpIt, sourceTensor, om, opIt);
    if (outputMemoryLocation.isForced())
        opIt->getOutputTensor(0)->set<mv::Tensor::MemoryLocation>("Location", outputMemoryLocation);
}
