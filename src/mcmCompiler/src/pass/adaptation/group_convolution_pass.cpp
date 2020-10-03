#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include <cmath>
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/custom_strings.hpp"
#include "include/mcm/tensor/quantization_params.hpp"
#include "mcm/utils/custom_math.hpp"
#include <numeric>
#include <cmath>

static void handleGroupConvolutionFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&);

namespace mv
{

    namespace pass
    {

        MV_REGISTER_PASS(HandleGroupConvolution)
        .setFunc(handleGroupConvolutionFcn)
        .setDescription(
            "Replaces group convolution"
        );
    }

}

void handleGroupConvolutionFcn(const mv::pass::PassEntry&, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
{
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)
    mv::OpModel om(model);
    mv::DataModel dm(model);
    auto convOps = om.getOps("Conv");

    for (auto& convOp : convOps)
    {
        auto group = convOp->get<unsigned>("group");
        if (group > 1)
        {
            auto inputTensor = convOp->getInputTensor(0);
            auto outputTensor = convOp->getOutputTensor(0);
            auto previousOp = om.getSourceOp(inputTensor);
            auto weightTensor = convOp->getInputTensor(1);
            auto inputChannels = inputTensor->getShape()[mv::IO_CHANNEL_DIMENSION];
            auto outputChannels = outputTensor->getShape()[mv::IO_CHANNEL_DIMENSION];

            auto inputGroupSize = inputChannels/group;
            auto weightsGroupSize = outputChannels/group;

            std::vector<mv::Data::OpListIterator> sinkOperators = findSinkLayers(dm, outputTensor);
            mv::Shape groupShape = {inputTensor->getShape()[mv::IO_WIDTH_DIMENSION],
                                   inputTensor->getShape()[mv::IO_HEIGHT_DIMENSION],
                                   inputGroupSize, 1};

            mv::Shape weightsGroupShape = {weightTensor->getShape()[mv::KERNEL_WIDTH],
                                   weightTensor->getShape()[mv::KERNEL_HEIGHT],
                                   weightTensor->getShape()[mv::KERNEL_INPUT_CHANNELS], weightsGroupSize};
            mv::Shape groupBegin = {{0},{0},{0},{0}};
            mv::Shape weightsGroupBegin = {{0},{0},{0},{0}};
            std::vector< mv::Data::TensorIterator> convOutputs = {};
            mv::Data::TensorIterator biasTensor;
            mv::QuantizationParams inputQuantParams  = inputTensor->getQuantParams();
            mv::QuantizationParams weightQuantParams = weightTensor->getQuantParams();
            mv::QuantizationParams outputQuantParams = outputTensor->getQuantParams();
            if (convOp->hasAttr("bias"))
                biasTensor = dm.getTensor(convOp->get<std::string>("bias"));
            for (unsigned branchId = 0; branchId < group; branchId++)
            {
                std::string sliceName = convOp->getName() + "slice" + std::to_string(branchId);
                std::string weightSliceName = convOp->getName() + "weightSlice" + std::to_string(branchId);
                std::string convName = convOp->getName() + "_" + sliceName;
                std::string biasName = mv::createBiasName(convName + "bias");
                groupBegin = {{0},{0},{branchId * inputGroupSize},{0}};
                auto slice = om.slice(sliceName,
                                      inputTensor,
                                      groupBegin,
                                      groupShape);
                slice->setQuantParams(inputQuantParams);

                om.getSourceOp(slice)->set<unsigned>("opId", convOp->get<unsigned>("opId"));

                weightsGroupBegin = {{0},{0},{0},{branchId * weightsGroupSize}};

                auto weightsSlice = om.slice(weightSliceName,
                                             weightTensor,
                                             weightsGroupBegin,
                                             weightsGroupShape);
                weightsSlice->setQuantParams(weightQuantParams);

                om.getSourceOp(weightsSlice)->set<unsigned>("opId", om.getSourceOp(convOp->getInputTensor(1))->get<unsigned>("opId"));

                auto newConvTensor = om.conv(convName,
                                             slice,
                                             weightsSlice,
                                             convOp->get<std::array<unsigned short, 2>>("stride"),
                                             convOp->get("padding"),
                                             convOp->get<unsigned>("dilationFactor"),
                                             1);
                newConvTensor->setDType(convOp->getOutputTensor(0)->getDType());
                newConvTensor->setQuantParams(outputQuantParams);
                om.getSourceOp(newConvTensor)->set<unsigned>("opId", convOp->get<unsigned>("opId"));
                auto sliceConvOp = om.getSourceOp(newConvTensor);
                if (convOp->hasAttr("bias"))
                {
                    mv::Data::TensorIterator biasSliceTensor;
                    std::vector<mv::DataElement> biasData;
                    for (std::size_t i = branchId * outputChannels; i < branchId * outputChannels + outputChannels; i++)
                        biasData.push_back(biasTensor->getData()[i]);

                    biasSliceTensor = dm.defineTensor(mv::Tensor(biasName + "slice" + std::to_string(branchId), {outputChannels}, biasTensor->getDType(),
                                        biasTensor->getOrder(), biasData, biasTensor->get<mv::QuantizationParams>("quantParams")));
                    om.addAttr(sliceConvOp, "bias", biasSliceTensor->getName());
                }
                convOutputs.push_back(newConvTensor);
            }
            auto concat = om.concat(convOp->getName() + "concat_",
                            convOutputs,
                            "C");
            concat->setDType(convOp->getOutputTensor(0)->getDType());
            concat->setQuantParams(outputQuantParams);
            om.getSourceOp(concat)->set<unsigned>("opId", convOp->get<unsigned>("opId"));
            for (auto sourceFlow = convOp.leftmostInput(); sourceFlow != om.flowEnd(); ++sourceFlow)
            {
                if (sourceFlow.source()->getName() == previousOp->getName())
                    om.undefineFlow(sourceFlow);
            }
            // Update all the sink operations input tensor to newly created concat output
            for (auto& sinkOp : sinkOperators)
            {
                sinkOp->setInputTensor(concat, 0, false);
                om.defineFlow(concat, sinkOp, 0);
            }
            om.removeOp(convOp);
        }
        else
            continue;
    }
}
