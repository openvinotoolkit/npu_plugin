#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/pass/pass_quantization.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/pass/pass_utils.hpp"
#include "include/mcm/utils/custom_math.hpp"

static void ensureDTypeCompatibilityFcn(const mv::pass::PassEntry& pass, mv::ComputationModel& model,
                                        mv::TargetDescriptor& targetDesc, mv::Element&, mv::Element&);

namespace mv {

namespace pass {

MV_REGISTER_PASS(EnsureDTypeCompatibility)
        .setFunc(ensureDTypeCompatibilityFcn)
        .setDescription("Ensure all HW tasks have compatible dtype combinationsm. "
                        "Apply the right mitigations taking into account performance and accuracy");

}

}  // namespace mv

// Most KMB dtype restrictions defined in the HW doc are related to the datatype that's
// resolved at MPE stage.
// Eltwise is different in the way that it bypasses the MPE and it goes directly to PPE
// Even if eltwise uses the same IDU unit dedicated to weights(for the read of the
// second input tensor) it does not inherit any mix datatype support or restrictions from the
// cases that use MPE.
// In short eltwise is forced to have the same dtype for its input tensors.

const std::vector<std::pair<mv::Data::TensorIterator, std::string>> dataTypeDiscrepancySolver(
        mv::OpModel& om, mv::Data::OpListIterator& opIt, mv::DataTypeSupport& dtypeCase) {
    using tensorSolverFunc =
            std::function<mv::Data::TensorIterator(const mv::Data::OpListIterator&, const mv::OpModel&)>;
    const std::unordered_map<std::string, tensorSolverFunc> tensorSolverMap = {
            {"input_0",
             [](const mv::Data::OpListIterator& op, const mv::OpModel&) {
                 return op->getInputTensor(0);
             }},
            {"input_1",
             [](const mv::Data::OpListIterator& op, const mv::OpModel& opModel) {
                 if (op->getOpType() == "Eltwise" ||
                     (op->getOpType() == "DPUTask" && op->get<std::string>("taskOp") == "Eltwise"))
                     return op->getInputTensor(1);
                 return opModel.tensorEnd();
             }},
            {"weights",
             [](const mv::Data::OpListIterator& op, const mv::OpModel& opModel) {
                 if (op->hasWeights())
                     return op->getInputTensor(1);
                 return opModel.tensorEnd();
             }},
            {"output", [](const mv::Data::OpListIterator& op, const mv::OpModel&) {
                 return op->getOutputTensor(0);
             }}};

    auto mitigationPlan = std::vector<std::pair<mv::Data::TensorIterator, std::string>>();
    auto isFail = true;
    for (auto failCaseEntry : dtypeCase.failCase) {
        auto tensorSolver = tensorSolverMap.find(failCaseEntry.first);
        if (tensorSolver == tensorSolverMap.cend())
            throw mv::RuntimeError(
                    om, opIt->getName() + ": No tensor dtype solver registered for dtype " + failCaseEntry.first);
        auto tensor = tensorSolver->second(opIt, om);
        if (tensor == om.tensorEnd() || tensor->getDType().toString() != failCaseEntry.second)
            isFail = false;
    }
    if (isFail) {
        for (auto mitigationEntry : dtypeCase.mitigation) {
            auto tensor = tensorSolverMap.find(mitigationEntry.first)->second(opIt, om);
            if (tensor != om.tensorEnd() && tensor->getDType().toString() != mitigationEntry.second)
                mitigationPlan.push_back({tensor, mitigationEntry.second});
        }
    }
    return mitigationPlan;
}

using dtypeConversionFunc =
        std::function<mv::Data::TensorIterator(mv::Data::TensorIterator&, mv::Data::OpListIterator&, mv::OpModel&)>;

mv::Data::TensorIterator convertFP16ToU8(mv::Data::TensorIterator& tensorIt, mv::Data::OpListIterator& opIt,
                                         mv::OpModel& opModel) {
    auto quantTensor = opModel.tensorEnd();
    if (tensorIt->isPopulated()) {
        // For both fast access by avoiding data transposing
        // and ease of data interation set order to channel major
        // of arbitrary dimensions
        auto tensorShape = tensorIt->getShape();
        auto backupOrder = tensorIt->getOrder();
        tensorIt->setOrder(mv::Order::getColMajorID(tensorShape.ndims()));

        auto outputTensor = opIt->getOutputTensor(0);
        auto numChannels = outputTensor->getShape()[mv::IO_CHANNEL_DIMENSION];
        auto wSetSize = tensorShape.totalSize() / numChannels;

        auto fp16TensorData = tensorIt->getData();
        auto floatTensorData = std::vector<double>(fp16TensorData.size());
        std::transform(fp16TensorData.cbegin(), fp16TensorData.cend(), floatTensorData.begin(), [](const int64_t& arg) {
            return mv::fp16_to_fp32(arg);
        });

        // A lot of compiler logic aligns weights and quant buffers to match
        // HW restrictions of 16 multiple out channels
        size_t numChannelsToCompute = numChannels;
        if (outputTensor->hasAttr("oldDimensions"))
            numChannelsToCompute = outputTensor->get<mv::Shape>("oldDimensions")[mv::IO_CHANNEL_DIMENSION];

        if (numChannelsToCompute > numChannels)
            throw mv::RuntimeError("DType Compatibility ", opIt->getName() + ": More channels require computation than actual channels " +
                            std::to_string(numChannelsToCompute) + " vs " + std::to_string(numChannels) + " for populated tensor " + tensorIt->getName());

        // Step 1 compute optimal quantization for weights tensor
        auto minRange = std::vector<double>(numChannelsToCompute);
        auto maxRange = std::vector<double>(numChannelsToCompute);
        for (std::size_t chIdx = 0; chIdx < numChannelsToCompute; chIdx++) {
            auto minMax = std::minmax_element(floatTensorData.cbegin() + chIdx * wSetSize,
                                              floatTensorData.cbegin() + (chIdx + 1) * wSetSize);
            minRange[chIdx] = *minMax.first;
            maxRange[chIdx] = *minMax.second;
        }
        auto scale = std::vector<double>(numChannelsToCompute);
        auto zp = std::vector<int64_t>(numChannelsToCompute);

        // KMB HW supports only per channel weights scale
        // ZP is per tensor value for all weights
        // In that case need to adapt quantization ranges
        // so that zero point is aligned
        calcAlignedZeroPointAndScalePerChannel(maxRange, minRange, 256, mv::getDType(mv::Precision::U8), scale, zp);

        // Step 2 quantized the float weights data
        auto quantTensorData = std::vector<int64_t>(floatTensorData.size());
        for (std::size_t chIdx = 0; chIdx < numChannelsToCompute; chIdx++) {
            auto chScale = scale.at(chIdx);
            auto min = minRange.at(chIdx);
            std::transform(floatTensorData.cbegin() + chIdx * wSetSize,
                           floatTensorData.cbegin() + (chIdx + 1) * wSetSize,
                           quantTensorData.begin() + chIdx * wSetSize, [chScale, min](const double& floatVal) {
                               return std::round((floatVal - min) / chScale);
                           });
        }

        // Step 2.1 resize quant vectors to padded size
        if (numChannels != numChannelsToCompute) {
            zp.resize(numChannels, 0);
            scale.resize(numChannels, 1);
            minRange.resize(numChannels, 0);
            maxRange.resize(numChannels, 0);
        }

        // Step 3 create new constantInt op and link correctly
        auto sourceFloatOp = opModel.getSourceOp(tensorIt);
        auto attrsToCopy = tensorIt->getAttrs({"dType", "Shape", "order", "sourceOp", "flows", "quantParams"});
        auto quantConst = opModel.constantInt(sourceFloatOp->getName() + "_quant_mitigated", quantTensorData,
                                              tensorIt->getShape(), mv::DType("UInt8"), tensorIt->getOrder());
        opModel.getSourceOp(quantConst)->set<unsigned>("opId", sourceFloatOp->get<unsigned>("opId"));
        quantConst->setQuantParams({{zp}, {scale}, {minRange}, {maxRange}});
        quantConst->setAttrs(attrsToCopy);

        linkNewOperationsRemove(opModel.getSourceOp(quantConst), opModel.tensorEnd(), opModel, sourceFloatOp);

        quantConst->setOrder(backupOrder);
        opModel.getSourceOp(quantConst)->set<mv::Order>("order", backupOrder);

        quantTensor = quantConst;
    } else {
        throw mv::RuntimeError("DType Compatibility ",
                               opIt->getName() + ": No dtype Float16 -> UInt8 conversion registered for " +
                                       "unpopulated tensor " + tensorIt->getName());
    }
    return quantTensor;
}

mv::Data::TensorIterator convertU8ToI8(mv::Data::TensorIterator& tensorIt, mv::Data::OpListIterator& opIt,
                                       mv::OpModel& opModel) {
    auto requantTensor = opModel.tensorEnd();
    if (tensorIt->isPopulated()) {
        // For both fast access by avoiding data transposing
        // and ease of data interation set order to channel major
        // of arbitrary dimensions
        auto tensorShape = tensorIt->getShape();
        auto backupOrder = tensorIt->getOrder();
        tensorIt->setOrder(mv::Order::getColMajorID(tensorShape.ndims()));
        auto targetDType = mv::DType("Int8");

        // Step 1 extend min max ranges so they are symmetrical
        auto quantParams = tensorIt->getQuantParams();
        auto outputTensor = opIt->getOutputTensor(0);
        auto numChannels = outputTensor->getShape()[mv::IO_CHANNEL_DIMENSION];
        auto wSetSize = tensorShape.totalSize() / numChannels;

        // Step 1.1 reduce the logic to be performed only on the channels used for compute
        // A lot of compiler logic aligns weights and quant buffers to match
        // HW restrictions of 16 multiple out channels
        size_t numChannelsToCompute = numChannels;
        if (outputTensor->hasAttr("oldDimensions"))
            numChannelsToCompute = outputTensor->get<mv::Shape>("oldDimensions")[mv::IO_CHANNEL_DIMENSION];

        auto minRange = quantParams.getMin();
        auto maxRange = quantParams.getMax();
        auto symmetricMinRange = std::vector<double>(numChannelsToCompute);
        auto symmetricMaxRange = std::vector<double>(numChannelsToCompute);

        std::transform(minRange.cbegin(), minRange.cbegin() + numChannelsToCompute,
            maxRange.cbegin(), symmetricMaxRange.begin(),
            [](const double& min, const double& max) {
                return std::max(abs(min), abs(max));
            });
        std::transform(symmetricMaxRange.cbegin(), symmetricMaxRange.cend(),
            symmetricMinRange.begin(),[](const double& max) {
                return -max;
            });

        auto symmetricRangeScale = std::vector<double>(numChannelsToCompute);
        mv::calcScalePerChannel(symmetricMaxRange, symmetricMinRange, 256, symmetricRangeScale);

        // Step 2 recompute weights given new symmetric quant scale
        auto tensorData = tensorIt->getIntData();
        auto I8TensorData = std::vector<int64_t>(tensorData.size());

        auto asymmZP = quantParams.getZeroPoint();
        auto asymmScale = quantParams.getScale();
        for (std::size_t chIdx = 0; chIdx < numChannelsToCompute; chIdx++) {
            auto zp = asymmZP.at(chIdx);
            auto rescale = asymmScale.at(chIdx) / symmetricRangeScale.at(chIdx);
            std::transform(tensorData.cbegin() + wSetSize * chIdx,
                tensorData.cbegin() + wSetSize * (chIdx + 1),
                I8TensorData.begin() + wSetSize * chIdx,
                [rescale, zp] (const int64_t& val) {
                    return std::round(static_cast<double>(val - zp) * rescale);
                });
        }

        // Step 3 update quantization params info
        auto symmetricZp = std::vector<int64_t>(numChannels, 0);
        // Step 3.1 resize quant vectors to padded size
        if (numChannels != numChannelsToCompute) {
            symmetricRangeScale.resize(numChannels, 1);
            symmetricMinRange.resize(numChannels, 0);
            symmetricMaxRange.resize(numChannels, 0);
        }

        quantParams = mv::QuantizationParams(symmetricZp, symmetricRangeScale,
            symmetricMinRange, symmetricMaxRange);

        // Step 4 create new constantInt op due to Int8 internal representation
        // and link correctly
        auto sourceIntOp = opModel.getSourceOp(tensorIt);
        auto attrsToCopy = tensorIt->getAttrs({"dType", "Shape", "order", "sourceOp", "flows", "quantParams"});
        auto I8Const = opModel.constantInt(sourceIntOp->getName() + "_symmetric_range_mitigated", I8TensorData,
                                           tensorIt->getShape(), targetDType, tensorIt->getOrder());
        opModel.getSourceOp(I8Const)->set<unsigned>("opId", sourceIntOp->get<unsigned>("opId"));
        I8Const->setAttrs(attrsToCopy);
        I8Const->setQuantParams(quantParams);
        I8Const->setDType(targetDType);
        linkNewOperationsRemove(opModel.getSourceOp(I8Const), opModel.tensorEnd(), opModel, sourceIntOp);

        I8Const->setOrder(backupOrder);
        opModel.getSourceOp(I8Const)->set<mv::Order>("order", backupOrder);

        requantTensor = I8Const;
    } else {
        throw mv::RuntimeError("DType Compatibility ", opIt->getName() +
                                                               ": No dtype UInt8 -> Int8 conversion registered for " +
                                                               "unpopulated tensor " + tensorIt->getName());
    }
    return requantTensor;
}

mv::Data::TensorIterator convertU8ToFP16(mv::Data::TensorIterator& tensorIt, mv::Data::OpListIterator& opIt,
                                         mv::OpModel& opModel) {
    auto dequantTensor = opModel.tensorEnd();
    if (tensorIt->isPopulated()) {
        // For both fast access by avoiding data transposing
        // and ease of data interation set order to channel major
        // of arbitrary dimensions
        auto tensorShape = tensorIt->getShape();
        auto backupOrder = tensorIt->getOrder();
        tensorIt->setOrder(mv::Order::getColMajorID(tensorShape.ndims()));

        auto numChannels = opIt->getOutputTensor(0)->getShape()[mv::IO_CHANNEL_DIMENSION];
        auto wSetSize = tensorShape.totalSize() / numChannels;

        auto quantTensorData = tensorIt->getIntData();
        auto quantParams = tensorIt->getQuantParams();

        // Step 1 dequantized the weights data
        auto dequantTensorData = std::vector<double>(quantTensorData.size());
        for (std::size_t chIdx = 0; chIdx < numChannels; chIdx++) {
            auto chScale = quantParams.getScale(chIdx);
            auto chZp = quantParams.getZeroPoint(chIdx);
            std::transform(quantTensorData.cbegin() + chIdx * wSetSize,
                           quantTensorData.cbegin() + (chIdx + 1) * wSetSize,
                           dequantTensorData.begin() + chIdx * wSetSize, [chScale, chZp](const int64_t& intVal) {
                               return static_cast<double>(intVal - chZp) * chScale;
                           });
        }

        // Step 2 explicit conversion to fp16
        auto dequantFP16TensorData = std::vector<int64_t>(dequantTensorData.size());
        std::transform(dequantTensorData.cbegin(), dequantTensorData.cend(), dequantFP16TensorData.begin(),
                       [](const double& floatVal) {
                           return mv::fp32_to_fp16(floatVal);
                       });

        // Step 3 create new constantInt op due to FP16 internal representation
        // and link correctly
        auto sourceIntOp = opModel.getSourceOp(tensorIt);
        auto attrsToCopy = tensorIt->getAttrs({"dType", "Shape", "order", "sourceOp", "flows", "quantParams"});
        auto dequantFP16Const =
                opModel.constantInt(sourceIntOp->getName() + "_dequant_mitigated", dequantFP16TensorData,
                                    tensorIt->getShape(), mv::DType("Float16"), tensorIt->getOrder());
        opModel.getSourceOp(dequantFP16Const)->set<unsigned>("opId", sourceIntOp->get<unsigned>("opId"));
        dequantFP16Const->setAttrs(attrsToCopy);

        linkNewOperationsRemove(opModel.getSourceOp(dequantFP16Const), opModel.tensorEnd(), opModel, sourceIntOp);

        dequantFP16Const->setOrder(backupOrder);
        opModel.getSourceOp(dequantFP16Const)->set<mv::Order>("order", backupOrder);
        dequantTensor = dequantFP16Const;
    } else {
        throw mv::RuntimeError("DType Compatibility ",
                               opIt->getName() + ": No dtype UInt8 -> Float16 conversion registered for " +
                                       "unpopulated tensor " + tensorIt->getName());
    }
    return dequantTensor;
}

static void ensureDTypeCompatibilityFcn(const mv::pass::PassEntry&, mv::ComputationModel& model,
                                        mv::TargetDescriptor& targetDesc, mv::Element&, mv::Element&) {
    MV_PROFILED_FUNCTION(MV_PROFILE_PASS)

    mv::OpModel om(model);
    mv::DataModel dm(model);

    // TODO: dtype conversions will behave differently for populated and unpopulated tensors
    // also need to determine of the tensor is input or output of current invetigated op
    // for either input or output flow, placinng a conversion layer will differ.
    const std::unordered_map<std::string, const std::unordered_map<std::string, dtypeConversionFunc>>
            dtypeConversionMap = {{"Float16", {{"UInt8", convertFP16ToU8}}},
                                  {"UInt8", {{"Int8", convertU8ToI8}}},
                                  {"UInt8", {{"Float16", convertU8ToFP16}}}};
    auto ops = om.getOps("DPUTask");
    for (auto opIt : ops) {
        for (auto dtypeCase : targetDesc.dtypeSupport()) {
            auto mitigationPlan = dataTypeDiscrepancySolver(om, opIt, dtypeCase);

            for (auto mitigationStep : mitigationPlan) {
                auto tensorIt = mitigationStep.first;
                auto targetDtype = mitigationStep.second;

                auto subMapEntry = dtypeConversionMap.find(tensorIt->getDType().toString());

                if (subMapEntry == dtypeConversionMap.cend())
                    throw mv::RuntimeError(om, tensorIt->getName() +
                                                       ": No dtype conversion registered for source dtype " +
                                                       tensorIt->getDType().toString());

                auto conversionFunctor = subMapEntry->second.find(targetDtype);

                if (conversionFunctor == subMapEntry->second.cend())
                    throw mv::RuntimeError(
                            om, tensorIt->getName() + ": No dtype conversion registered for target dtype " +
                                        tensorIt->getDType().toString() + " from source dtype " + targetDtype);

                conversionFunctor->second(tensorIt, opIt, om);
            }
        }
    }
}
