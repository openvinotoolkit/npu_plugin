#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/utils/custom_math.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/pass/pass_quantization.hpp"

namespace
{
static void collapseBranchedScaleFcn(
    const mv::pass::PassEntry& pass,
    mv::ComputationModel& model,
    mv::TargetDescriptor&, mv::Element&, mv::Element&);
}

namespace mv
{
    namespace pass
    {
        MV_REGISTER_PASS(CollapseBranchedScale)
            .setFunc(collapseBranchedScaleFcn)
            .setDescription(
                "Fuse branched scale ops and collapse the braching"
            );
    }
}

// For an example like the one below
//..........
//     |
// [ Conv 32 ch]
//     |
//     |
// [ Bias 32 ch]_______
//     |               |
//     |               |
//     |            [ Scale 32 ch]
//     |               |
// [ Concat 64 ch]_____|
//     |
//     |
// [ Relu 64 ch]
//     |
//     |
// [ Fake Quantize 64 ch]
//     |
//..........
// We are incapable of duplicating output activations
// and applying a selective scale purely in HW.
// One solution would be doing mulitple DPU ops with FP16 precision
// but in this pass we aim to fuse the branched scale operation in the
// convs weights and biases increading the number of output channels but
// doing this as a single DPU op.
//..........
//     |
// [ Conv 64 ch]
//     |
//     |
// [ Bias 64 ch]
//     |
//     |
// [ Relu 64 ch]
//     |
//     |
// [ Fake Quantize 64 ch]
//     |
//..........

namespace
{

    mv::Data::OpListIterator findConcatSink(
        mv::Data::OpListIterator& op,
        mv::OpModel& om,
        const std::vector<std::string>& friendlyOpTypes)
    {
        std::set<std::string> concatSinksName;
        for (auto childOp = op.leftmostChild(); childOp != om.opEnd(); ++childOp) {
            auto opItr = mv::Data::OpListIterator(childOp);
            while (opItr->getOpType() != "Concat") {
                if (std::find(friendlyOpTypes.begin(), friendlyOpTypes.end(), opItr->getOpType())
                    != friendlyOpTypes.end() && opItr.childrenSize() == 1)
                    opItr = opItr.leftmostChild();
                else
                    return om.opEnd();
            }
            concatSinksName.insert(opItr->getName());
        }
        if (concatSinksName.size() == 1)
            return om.getOp(*concatSinksName.begin());
        return om.opEnd();
    }

    std::vector<mv::Data::OpListIterator> matchFusableParentPattern(
        mv::Data::OpListIterator& parentOp,
        mv::OpModel& om,
        const std::vector<std::vector<std::string>>& fusablePatterns)
    {

        const std::vector<std::string> constOps = {
            "Constant", "ConstantInt", "ConstantDataElement",
            "WeightsTable", "SparsityMap"};

        auto fusableOpChain = std::vector<mv::Data::OpListIterator>();
        for (auto pattern : fusablePatterns)
        {
            auto opItr = mv::Data::OpListIterator(parentOp);
            for (auto patternEntry : pattern)
            {
                if(opItr->getOpType() != patternEntry) {
                    fusableOpChain.clear();
                    break;
                }
                fusableOpChain.push_back(opItr);
                auto nextParent = opItr.leftmostParent();
                while (std::find(constOps.begin(), constOps.end(), nextParent->getOpType())
                    != constOps.end() && nextParent != om.opEnd())
                    ++ nextParent;

                if (nextParent == om.opEnd()) {
                    fusableOpChain.clear();
                    break;
                }
                opItr = nextParent;
            }
            if(fusableOpChain.size() == pattern.size())
                return fusableOpChain;
            fusableOpChain.clear();
        }

        return fusableOpChain;
    }

    template <class t>
    void resizeVectorExtendValues(std::vector<t> &vec, std::size_t size) {
        if (vec.size() == size)
            return;
        vec.resize(size, vec.back());
    }

    void fuseBranchedScales(const mv::pass::PassEntry&, mv::OpModel& om)
    {
        const std::vector<std::string> friendlyNeighborOps = {"Scale", "FakeQuantize"};
        const std::vector<std::vector<std::string>> fusableParentPatterns =
            {
                {"Conv"},
                {"Bias", "Conv"},
            };

        using fuseFunc = std::function<mv::Data::OpListIterator(const mv::Data::OpListIterator& scaleOp,
            mv::Data::OpListIterator& fuseOp,
            mv::OpModel& om)>;

        const std::unordered_map<std::string, fuseFunc> fuseFunctorMap =
        {
            {
                "Bias",
                [] (const mv::Data::OpListIterator& scaleOp,
                mv::Data::OpListIterator& fuseOp,
                mv::OpModel& opModel) {

                    auto inputTensor = fuseOp->getInputTensor(0);
                    auto biasConstTensor = fuseOp->getInputTensor(1);
                    auto biasConstOp = opModel.getSourceOp(biasConstTensor);

                    auto scaleData = scaleOp->getInputTensor(1)->getDoubleData();

                    auto biasData = biasConstTensor->getDoubleData();
                    auto scaledBiasData = std::vector<double>(biasData.size() * 2);
                    std::copy_n(biasData.cbegin(), biasData.size(), scaledBiasData.begin());

                    std::transform(biasData.cbegin(), biasData.cend(),
                        scaleData.cbegin(), scaledBiasData.begin() + biasData.size(),
                        std::multiplies<double>());

                    auto scaledBiasConstTensor = opModel.constant(
                        biasConstOp->getName() + "_fused_scale",
                        scaledBiasData,
                        {scaledBiasData.size()},
                        biasConstTensor->getDType(),
                        biasConstTensor->getOrder());
                    scaledBiasConstTensor->setQuantParams(biasConstTensor->getQuantParams());
                    opModel.getSourceOp(scaledBiasConstTensor)->set<unsigned>("opId",
                        biasConstOp->get<unsigned>("opId"));

                    auto scaledBiasTensor = opModel.bias(
                        fuseOp->getName() + "_fused_scale",
                        inputTensor,
                        scaledBiasConstTensor);
                    scaledBiasTensor->setQuantParams(fuseOp->getOutputTensor(0)->getQuantParams());
                    scaledBiasTensor->setDType(fuseOp->getOutputTensor(0)->getDType());
                    opModel.getSourceOp(scaledBiasTensor)->set<unsigned>("opId",
                        fuseOp->get<unsigned>("opId"));

                    linkNewMultipleOperationsReplacement(
                        opModel.getSourceOp(inputTensor), {scaledBiasTensor}, opModel, fuseOp);

                    return opModel.getSourceOp(scaledBiasTensor);
                }
            },
            {
                "Conv",
                [] (const mv::Data::OpListIterator& scaleOp,
                mv::Data::OpListIterator& fuseOp,
                mv::OpModel& opModel) {

                    auto inputTensor = fuseOp->getInputTensor(0);

                    auto wtOp = opModel.getSourceOp(fuseOp->getInputTensor(1));
                    auto wtFqOp = opModel.opEnd();
                    if (wtOp->getOpType() == "FakeQuantize") {
                        wtFqOp = wtOp;
                        wtOp = opModel.getSourceOp(wtFqOp->getInputTensor(0));
                    }

                    auto wtTensor = wtOp->getOutputTensor(0);
                    auto wtData = wtTensor->getDoubleData();
                    auto scaleData = scaleOp->getInputTensor(1)->getDoubleData();

                    auto scaledWt = std::vector<double>(wtData.size() * 2, 1.0);
                    auto wtSetSize = wtTensor->getShape()[mv::KERNEL_INPUT_CHANNELS] *
                        wtTensor->getShape()[mv::KERNEL_HEIGHT] *
                        wtTensor->getShape()[mv::KERNEL_WIDTH];
                    auto tensorDim = wtTensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS];

                    auto wtTensorFunctor = [](
                        mv::OpModel & opModelRef,
                        mv::Data::TensorIterator weightsTensor,
                        mv::Data::OpListIterator weightsOp,
                        std::vector<double> &scaledWeights) {
                        auto scaledWtShape = weightsTensor->getShape();
                        scaledWtShape[mv::KERNEL_OUTPUT_CHANNELS] *= 2;
                        auto newWeightsTensor = opModelRef.constant(
                            weightsOp->getName() + "_fused_scale",
                            scaledWeights,
                            scaledWtShape,
                            weightsTensor->getDType(),
                            weightsTensor->getOrder());
                        newWeightsTensor->setQuantParams(weightsTensor->getQuantParams());
                        opModelRef.getSourceOp(newWeightsTensor)->set<unsigned>("opId",
                            weightsOp->get<unsigned>("opId"));
                        return newWeightsTensor;
                        };

                    // Step 3.0 in the case of a FQ around weights
                    // need to recompute the new optimal quantization ranges
                    // and replace the FQ op and parent cont ops with the
                    // new variants
                    auto newWeightsTensor = opModel.tensorEnd();
                    if (wtFqOp != opModel.opEnd()) {

                        auto quantParamsI = extractQuantParamsI(wtFqOp, false);
                        auto quantParamsO = extractQuantParamsO(wtFqOp, false);
                        auto levels = wtFqOp->get<unsigned>("levels");

                        if (quantParamsO.isScalePerTensor())
                            throw mv::RuntimeError("CollapseBranchedScale",
                            "Working assumption: output quant params is per channel, "
                            "not satisfied");

                        auto scaleDataFit = std::vector<double>(tensorDim, scaleData[0]);
                        std::copy_n(scaleData.cbegin(),
                            std::min(tensorDim, scaleData.size()), scaleDataFit.begin());

                        // Step 3.0.1 calculate new extended pair of output quant ranges
                        // Compute the scaled FQ output quant range values
                        auto scaledQuantMinO = std::vector<double>(tensorDim);
                        auto quantMinO = quantParamsO.getMin();
                        std::transform(
                            quantMinO.cbegin(), quantMinO.cend(),
                            scaleDataFit.cbegin(), scaledQuantMinO.begin(),
                            std::multiplies<double>());

                        auto scaledQuantMaxO = std::vector<double>(tensorDim);
                        auto quantMaxO = quantParamsO.getMax();
                        std::transform(
                            quantMaxO.cbegin(), quantMaxO.cend(),
                            scaleDataFit.cbegin(), scaledQuantMaxO.begin(),
                            std::multiplies<double>());

                        // Compute final FQ output quant range values
                        // Taking into account new min/max result
                        auto newQuantMinO = std::vector<double>(tensorDim * 2);
                        std::copy_n(quantMinO.cbegin(), tensorDim, newQuantMinO.begin());
                        std::transform(quantMinO.cbegin(), quantMinO.cend(),
                            scaledQuantMinO.cbegin(), newQuantMinO.begin() + tensorDim,
                            [](double a, double b) {return std::min(a,b);});

                        auto newQuantMaxO = std::vector<double>(tensorDim * 2);
                        std::copy_n(quantMaxO.cbegin(), tensorDim, newQuantMaxO.begin());
                        std::transform(quantMaxO.cbegin(), quantMaxO.cend(),
                            scaledQuantMaxO.cbegin(), newQuantMaxO.begin() + tensorDim,
                            [](double a, double b) {return std::max(a,b);});

                        // Step 3.0.3 making use of the quantization scheme
                        // rescale the weights properly, by requantizing them
                        // into the output float range, applying the model scale
                        // and quantizing back with the updated FQ updated output
                        // range
                        // !!! Note that here we'd have a choice to also
                        // define a new FQ input range, but to avoid additional
                        // accuracy loss due to requantizing, we choose to simply use
                        // a dummy [0, levels] float range which will
                        // do a 1 to 1 mapping of the U8 values to input FP values
                        auto quantScaleI = quantParamsI.getScale();
                        resizeVectorExtendValues(quantScaleI, tensorDim);
                        auto quantScaleO = quantParamsO.getScale();
                        resizeVectorExtendValues(quantScaleO, tensorDim);

                        auto quantZpI = quantParamsI.getZeroPoint();
                        resizeVectorExtendValues(quantZpI, tensorDim);
                        auto quantZpO = quantParamsO.getZeroPoint();
                        resizeVectorExtendValues(quantZpO, tensorDim);

                        auto scaledWtNewWtItr = scaledWt.begin() + wtData.size();
                        for (size_t idx = 0; idx < tensorDim; ++idx)
                        {
                            auto scaleValue = scaleData.at(idx);

                            auto quantScaleICh = quantScaleI.at(idx);
                            auto quantScaleOCh = quantScaleO.at(idx);

                            auto quantZpICh = quantZpI.at(idx);
                            auto quantZpOCh = quantZpO.at(idx);

                            double scaledQuantScaleO = 1.0;
                            int64_t scaledQuantZpO = 0;
                            calcZeroPointAndScalePerTensor(
                                newQuantMaxO.at(idx + tensorDim),
                                newQuantMinO.at(idx + tensorDim),
                                scaledQuantScaleO,
                                scaledQuantZpO,
                                mv::getDType(mv::Precision::U8),
                                levels);

                            // To be able to use a dummy FQ input range
                            // of [0, levels] we must readapt both the original weights
                            // and the newly scaled weights
                            std::transform(
                                wtData.cbegin() + wtSetSize * idx,
                                wtData.cbegin() + wtSetSize * (idx + 1),
                                scaledWtNewWtItr + wtSetSize * idx,
                                [&]
                                (double e) {return std::round((e / quantScaleICh + quantZpICh - quantZpOCh)
                                * quantScaleOCh * scaleValue / scaledQuantScaleO + scaledQuantZpO);});

                            calcZeroPointAndScalePerTensor(
                                newQuantMaxO.at(idx),
                                newQuantMinO.at(idx),
                                scaledQuantScaleO,
                                scaledQuantZpO,
                                mv::getDType(mv::Precision::U8),
                                levels);
                            std::transform(
                                wtData.cbegin() + wtSetSize * idx,
                                wtData.cbegin() + wtSetSize * (idx + 1),
                                scaledWt.begin() + wtSetSize * idx,
                                [&]
                                (double e) {return std::round((e / quantScaleICh + quantZpICh - quantZpOCh)
                                * quantScaleOCh / scaledQuantScaleO + scaledQuantZpO);});
                        }
                        // Step 3.0.3 calculate new global min/max pair for FQ input range
                        // As noted previously, we use a dummy input quant range to bypass
                        // additional requantization loss
                        auto scaledQuantMinI = std::vector<double>
                            {0};
                        auto scaledQuantMaxI = std::vector<double>
                            {static_cast<double>(levels - 1)};
                        if (!quantParamsI.isScalePerTensor()) {
                            resizeVectorExtendValues(scaledQuantMinI, 2 * tensorDim);
                            resizeVectorExtendValues(scaledQuantMaxI, 2 * tensorDim);
                        }

                        // Step 3.0.4 re-instantiate all the FQ const ops
                        auto scaledQuantData = std::vector<std::pair<std::size_t, std::vector<double>&>>();
                        scaledQuantData.push_back({1, scaledQuantMinI});
                        scaledQuantData.push_back({2, scaledQuantMaxI});
                        scaledQuantData.push_back({3, newQuantMinO});
                        scaledQuantData.push_back({4, newQuantMaxO});

                        auto scaledTensors = std::vector<mv::Data::TensorIterator>();
                        for (auto scaledQuant : scaledQuantData) {
                            auto oldTensor = wtFqOp->getInputTensor(scaledQuant.first);
                            auto newTensorShape = oldTensor->getShape();
                            if (scaledQuant.second.size() == tensorDim * 2)
                                newTensorShape[mv::KERNEL_OUTPUT_CHANNELS] *= 2;
                            auto newTensor = opModel.constant(
                                    opModel.getSourceOp(oldTensor)->getName() + "_fused_scale",
                                    scaledQuant.second,
                                    newTensorShape,
                                    oldTensor->getDType(),
                                    oldTensor->getOrder());
                            newTensor->setQuantParams(oldTensor->getQuantParams());
                                opModel.getSourceOp(newTensor)->set<unsigned>("opId",
                                    opModel.getSourceOp(oldTensor)->get<unsigned>("opId"));
                            scaledTensors.push_back(newTensor);
                        }

                        // Step 3.0.5 re-instantiate the weights tensor and it's
                        // subsequent FQ op
                        auto scaledWeightsTensor = wtTensorFunctor(opModel, wtTensor, wtOp, scaledWt);
                        newWeightsTensor = opModel.fakeQuantize(
                            wtFqOp->getName() + "_fused_scale",
                            scaledWeightsTensor,
                            scaledTensors[0],
                            scaledTensors[1],
                            scaledTensors[2],
                            scaledTensors[3],
                            levels);
                        opModel.getSourceOp(newWeightsTensor)->set<unsigned>("opId",
                            wtFqOp->get<unsigned>("opId"));

                        linkNewOperationsRemove(opModel.getSourceOp(newWeightsTensor), opModel.tensorEnd(), opModel, wtFqOp);

                    } else {
                        // Step 3.1 Whenever no FQ is around, the solution reduces
                        // to simply enxtending and multiplying the weight with
                        // the model scale
                        for (size_t idx = 0; idx < wtTensor->getShape()[mv::KERNEL_OUTPUT_CHANNELS]; ++idx)
                        {
                            auto scaleValue = scaleData.at(idx);
                            std::transform(
                                wtData.cbegin() + wtSetSize * idx,
                                wtData.cbegin() + wtSetSize * (idx + 1),
                                scaledWt.begin() + wtData.size() + wtSetSize * idx,
                                [scaleValue](double e) {return e * scaleValue;});
                        }
                        newWeightsTensor = wtTensorFunctor(opModel, wtTensor, wtOp, scaledWt);
                    }

                    // Step 3.2 Finally create the conv operation and replace the
                    // new scaled subgraph in the model
                    auto newConvTensor = opModel.conv(
                        fuseOp->getName() + "_fused_scale",
                        inputTensor,
                        newWeightsTensor,
                        fuseOp->get<std::array<unsigned short, 2>>("stride"),
                        fuseOp->get<std::array<unsigned short, 4>>("padding"),
                        fuseOp->get<unsigned>("dilationFactor"),
                        fuseOp->get<unsigned>("group"));
                    newConvTensor->setQuantParams(fuseOp->getOutputTensor(0)->getQuantParams());
                    newConvTensor->setDType(fuseOp->getOutputTensor(0)->getDType());
                    opModel.getSourceOp(newConvTensor)->set<unsigned>("opId",
                        fuseOp->get<unsigned>("opId"));

                    linkNewMultipleOperationsReplacement(
                        opModel.getSourceOp(inputTensor), {newConvTensor}, opModel, fuseOp);

                    return opModel.getSourceOp(newConvTensor);
                }
            }
        };

        auto scaleOps = om.getOps("Scale");
        for(auto scaleOp : scaleOps)
        {
            auto parentOp = om.getSourceOp(scaleOp->getInputTensor(0));

            // Assumption:
            // For now try to tackle the simple case of only 1 branch
            // paralel to the scale
            if (parentOp.childrenSize() != 2)
                continue;

            // Step 1
            // Scope out the fuse chain pattern and the associate ops
            // in that chain
            auto fusableOpChain = matchFusableParentPattern(parentOp, om, fusableParentPatterns);
            if (fusableOpChain.empty())
                continue;

            // Step 2
            // Find concat sink operation
            // Allow friendly tasks on either of the branches such as FQs
            // Later logic will attempt to concatenate them
            auto concatSinkOp = findConcatSink(parentOp, om, friendlyNeighborOps);
            if (concatSinkOp == om.opEnd())
                continue;

            // Step 3
            // Apply the upstream fusing of the scale op
            // using per opType functor defined in fuseFunctorMap
            auto fusedParentOpChain = std::vector<mv::Data::OpListIterator>();
            for (auto fusableOpItr = fusableOpChain.rbegin();
                fusableOpItr != fusableOpChain.rend(); ++fusableOpItr)
            {
                auto fusableOp = *fusableOpItr;

                auto fuseFunctor = fuseFunctorMap.find(
                    fusableOp->getOpType());

                if (fuseFunctor == fuseFunctorMap.cend())
                    throw mv::RuntimeError(om, scaleOp->getName() +
                        ": No scale fuse functor registered for opTYpe " +
                        fusableOp->getOpType());

                auto fusedOp = fuseFunctor->second(scaleOp, fusableOp, om);
                fusedParentOpChain.push_back(fusedOp);
            }

            // Step 4
            // Remove the scale op and collapse the branches
            // At need fuse the later FQ layers if present of either of branches
            removeOperation(om.tensorEnd(), om, scaleOp);
            linkNewOperationsRemove(fusedParentOpChain.back(), om.tensorEnd(), om, concatSinkOp);

        }
    }

    void collapseBranchedScaleFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model, mv::TargetDescriptor&, mv::Element&, mv::Element&)
    {
        MV_PROFILED_FUNCTION(MV_PROFILE_PASS);
        mv::OpModel om(model);
        fuseBranchedScales(pass, om);
    }
}
