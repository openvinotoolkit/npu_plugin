//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

//
// MatMul operation takes two tensors and performs usual matrix-matrix multiplication, matrix-vector multiplication or
// vector-matrix multiplication depending on argument shapes. Input tensors can have any rank >= 1. Two right-most axes
// in each tensor are interpreted as matrix rows and columns dimensions while all left-most axes (if present) are
// interpreted as multi-dimensional batch: [BATCH_DIM_1, BATCH_DIM_2,..., BATCH_DIM_K, ROW_INDEX_DIM, COL_INDEX_DIM].
// The operation supports usual broadcast semantics for batch dimensions. It enables multiplication of batch of pairs of
// matrices in a single shot. Before matrix multiplication, there is an implicit shape alignment for input arguments. It
// consists of the following steps:
// 1. Applying transpositions specified by optional transpose_a and transpose_b attributes. Only the two right-most
// dimensions are transposed, other dimensions remain the same. Transpose attributes are ignored for 1D tensors.
// 2. One-dimensional tensors unsqueezing is applied for each input independently. The axes inserted in this step are
// not included in the output shape.
//    - If rank of the first input is equal to 1, it is always unsqueezed to 2D tensor row vector (regardless of
//    transpose_a) by adding axes with size 1 at ROW_INDEX_DIM, to the left of the shape. For example [S] will be
//    reshaped to [1, S].
//    - If rank of the second input is equal to 1, it is always unsqueezed to 2D tensor column vector (regardless of
//    transpose_b) by adding axes with size 1 at COL_INDEX_DIM, to the right of the shape. For example [S] will be
//    reshaped to [S, 1].
//    - Temporary axes inserted in this step are removed from the final output shape after multiplying. After
//    vector-matrix multiplication, the temporary axis inserted at ROW_INDEX_DIM is removed. After matrix-vector
//    multiplication, the temporary axis inserted at COL_INDEX_DIM is removed. Output shape of two 1D tensors
//    multiplication [S] x [S] is squeezed to scalar.
// Example)
//   [M, K] * [K, N] => [M, N]
//   [K]    * [K, N] => [1, K] * [K, N] => [1, N] => [N]
//   [M, K] * [K]    => [M, K] * [K, 1] => [M, 1] => [M]
//   [M]    * [M]    => [1, M] * [M, 1] => [1, 1] => [1]
//
// 3. If ranks of input arguments are different after steps 1 and 2, the tensor with a smaller rank is unsqueezed from
// the left side of the shape by necessary number of axes to make both shapes of the same rank.
// 4. Usual rules of the broadcasting are applied for batch dimensions.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::MatMulOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueRange operands, mlir::DictionaryAttr attrs,
        mlir::RegionRange, SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::MatMulOpAdaptor matMul(operands, attrs);
    if (mlir::failed(matMul.verify(loc))) {
        return ::mlir::failure();
    }

    const auto& inType1 = matMul.input1().getType().cast<mlir::RankedTensorType>();
    const auto& inType2 = matMul.input2().getType().cast<mlir::RankedTensorType>();
    const auto& inShape1 = inType1.getShape();
    const auto& inShape2 = inType2.getShape();
    const auto& inRank1 = inShape1.size();
    const auto& inRank2 = inShape2.size();
    const auto transA = matMul.transpose_a() != nullptr;
    const auto transB = matMul.transpose_b() != nullptr;

    // Rightmost two axes are row & col. Remaining left axes are batch
    constexpr int kRowColIdxRange = 2;

    std::vector<int64_t> outShape;
    outShape.reserve(std::max(inRank1, inRank2));

    // Temporally transformed shapes
    std::vector<int64_t> inShape1Trans = inShape1;
    std::vector<int64_t> inShape2Trans = inShape2;
    std::reverse(inShape1Trans.begin(), inShape1Trans.end());
    std::reverse(inShape2Trans.begin(), inShape2Trans.end());

    // Apply transpose only when rank >= 2
    if (transA && (inRank1 > 1))
        std::swap(inShape1Trans[0], inShape1Trans[1]);
    if (transB && (inRank2 > 1))
        std::swap(inShape2Trans[0], inShape2Trans[1]);

    // Only use the dim when it is Mat
    if (inRank2 >= kRowColIdxRange)
        outShape.push_back(inShape2Trans[0]);
    if (inRank1 >= kRowColIdxRange)
        outShape.push_back(inShape1Trans[1]);

    // Process batch axes
    uint32_t idx1 = kRowColIdxRange;
    uint32_t idx2 = kRowColIdxRange;

    while (idx1 < inRank1 || idx2 < inRank2) {
        if (idx1 < inRank1 && idx2 < inRank2) {
            outShape.push_back(std::max(inShape1Trans[idx1], inShape2Trans[idx2]));
            ++idx1;
            ++idx2;
        } else if (idx2 >= inRank2) {
            outShape.push_back(inShape1Trans[idx1]);
            ++idx1;
        } else if (idx1 >= inRank1) {
            outShape.push_back(inShape2Trans[idx2]);
            ++idx2;
        }
    }
    std::reverse(std::begin(outShape), std::end(outShape));

    auto outType = mlir::RankedTensorType::get(outShape, inType1.getElementType());
    inferredReturnShapes.emplace_back(outType.getShape(), outType.getElementType());

    return mlir::success();
}
