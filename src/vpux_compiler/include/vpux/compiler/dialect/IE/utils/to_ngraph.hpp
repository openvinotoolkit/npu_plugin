//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/compiler/dialect/IE/attributes/enums.hpp"
#include "vpux/compiler/dialect/IE/attributes/structs.hpp"

#include <ngraph/opsets/opset7.hpp>

namespace opset_latest = ngraph::opset7;

namespace vpux {
namespace IE {

ngraph::op::AutoBroadcastType exportBroadcastType(AutoBroadcastType bType);
ngraph::op::BroadcastType exportBroadcastMode(BroadcastType bType);
ngraph::op::RoundingType exportRoundingType(RoundingType roundingType);
ngraph::element::Type exportElemType(mlir::MLIRContext* ctx, mlir::Type type);
ngraph::op::DetectionOutputAttrs exportDetectionOutputAttrs(const DetectionOutputAttr& val);
ngraph::opset7::Interpolate::InterpolateAttrs exportInterpolateAttrs(const InterpolateAttr& val);
std::string exportLRN_IERegion(LRN_IERegion region);
ngraph::op::RecurrentSequenceDirection exportRNNSequenceDirection(const RNNSequenceDirection val);
ngraph::op::TopKSortType exportTopKSortType(TopKSortType val);
ngraph::op::PadMode exportPadMode(PadMode mode);
ngraph::op::ProposalAttrs exportProposalAttrs(const ProposalAttr& val);
ngraph::op::v5::Round::RoundMode exportRoundMode(RoundMode val);
std::string exportROIPoolingMethod(ROIPoolingMethod method);
ngraph::op::TopKMode exportTopKMode(TopKMode val);
ngraph::op::TopKSortType exportTopKSortType(TopKSortType val);
InferenceEngine::TensorDesc exportUserTensor(const mlir::RankedTensorType& tensor);
ngraph::element::Type toNGraphType(const InferenceEngine::Precision& precision);
InferenceEngine::Precision exportPrecision(mlir::MLIRContext* ctx, mlir::Type type);

}  // namespace IE
}  // namespace vpux
