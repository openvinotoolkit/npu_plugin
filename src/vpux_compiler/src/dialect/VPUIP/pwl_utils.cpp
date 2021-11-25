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

#include "vpux/compiler/dialect/VPUIP/pwl_utils.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

/* The PWL table is currently only used for U8 computation.
 * When PWL is enabled, the hardware compute pipeline looks as follows:
 *
 *   Compute_stage -> [i32] -> Requant_stage * ((in_sc * wt_sc) / out_sc) -> CLAMP
 *          [i13] -> PWL -> [i13] -> PostShift_stage << >> -> [i8] -> + ZP -> [u8]
 *
 * The PWL table requires fixed input and output quantization ranges for i13 data.
 * For example, the pre-trained Sigmoid works best with [-4.0, 4.0] input range
 * and [0.0, 1.0] output range. To ensure the correct range reaches the PWL, we
 * need to map [-4.0, 4.0] to [-4096, 4095] (i13). This can be achieved in the
 * requantization stage by dropping the original operation's quantization and
 * enforcing the one needed by the PWL: setting the output scale (out_sc) to 1/1024
 * and the clamp values to [-4096, 4095].
 *
 * The output of the PWL table for the Sigmoid will contain values between [0, 4095],
 * which map to [0.0, 1.0] when dequantized. The post-shift stage helps translate the
 * i13 results to i8/u8. Therefore, [0, 4095] >> 4 results in [0, 255]. The zero-point
 * is then set to zero since the data is u8. The consumer operations need to account
 * for this enforced quantization as well, so zero-point 0 and scale 1/255 are used to
 * represent the [0.0, 1.0] float interval.
 *
 * The same behavior occurs for Tanh, with the mention that the output of the PWL table
 * will contain values between [-4096, 4095] (corresponding to float [-1, 1]). Applying
 * post-shift [-4096, 4095] >> 5 leads to [-128, 127], so a zero-point of 128 is added
 * to obtain u8 results.
 *
 * NOTE: The scale and zero-point values mentioned above are fine-tuned to best fit the
 * hardcoded PWL values for the Sigmoid and Tanh operations; they have been observed to
 * lead to the best average accuracy.
 */

const EnumMap<VPUIP::PPELayerType, VPUIP::PwlQuantReqs> vpux::VPUIP::pwlQuantReqs = {
        {VPUIP::PPELayerType::SIGMOID, {{-4.0, 4.0, 1.0 / 1015.6875, 0, 4}, {0.0, 1.0, 1.0 / 249.0, 3, 0}}},  //
        {VPUIP::PPELayerType::TANH, {{-4.0, 4.0, 1.0 / 903.5, 128, 5}, {-1.0, 1.0, 1.0 / 127.0, 128, 0}}},    //
};

VPUIP::PwlQuantReqs vpux::VPUIP::getPwlQuantReqs(const VPUIP::PPELayerType ppeType) {
    const auto quantReqs = pwlQuantReqs.find(ppeType);
    VPUX_THROW_UNLESS(quantReqs != pwlQuantReqs.cend(), "Missing quantization requirements for PWL post-op {0}",
                      ppeType);
    return quantReqs->second;
}

VPUIP::PwlQuantReqs* vpux::VPUIP::getCustomPwlQuantReqs(IE::LayerWithPostOpInterface origOp) {
    // const auto quantReqs = pwlQuantReqs.find(ppeType);
    // VPUX_THROW_UNLESS(quantReqs != pwlQuantReqs.cend(), "Missing quantization requirements for PWL post-op {0}",
    //                  ppeType);

    auto inputType = origOp->getOperand(0)
                             .getType()
                             .cast<mlir::RankedTensorType>()
                             .getElementType()
                             .dyn_cast<mlir::quant::UniformQuantizedType>();
    auto outputType = origOp->getResult(0)
                              .getType()
                              .cast<mlir::RankedTensorType>()
                              .getElementType()
                              .dyn_cast<mlir::quant::UniformQuantizedType>();

    PWLTableType pwl_type;
    pwl_type.activation = origOp.getPostOp().getValue().getStringRef().str();
    pwl_type.dtype = mlir::quant::UniformQuantizedType();
    PWLTableMap* pwl_table_map = customPWLTable_leakyRelu();
    auto pwl_table_vec = pwl_table_map->at(pwl_type);

    auto pwl_table_range = pwl_table_vec[0].range;
    auto pwl_range = std::make_pair(pwl_table_range[0], pwl_table_range[pwl_table_range.size() - 1]);

    float fl_min;
    float fl_max;
    int64_t levels;
    getFakeQuantParams(outputType, levels, fl_min, fl_max);
    double scale = outputType.getScale();
    double zeroPoint = outputType.getZeroPoint();
    if (scale == 0.0) {
        // error
    }
    int q_max = std::round(fl_max / scale + zeroPoint);

    double last_fl_max = fl_max;
    double last_q_max = pwl_range.second;
    // NOTE: zero point is added after lr so the formula should contain the subtraction
    bool leakyReluCase = false;

    if (origOp.getPostOpAttrs().get("negative_slope")) {
        double leakyAlpha = origOp.getPostOpAttrs().get("negative_slope").cast<mlir::FloatAttr>().getValueAsDouble();
        double fl_min_before_lrelu = fl_min / leakyAlpha;
        int q_min = std::round(fl_min_before_lrelu / scale + zeroPoint);
        const auto scaled_q_min = std::round(std::abs(q_min) * leakyAlpha);
        leakyReluCase = (scaled_q_min - zeroPoint) > std::abs(pwl_range.first);
    }

    if (q_max - zeroPoint > pwl_range.second || leakyReluCase) {
        int64_t newZp = zeroPoint;
        double newScale = last_fl_max / (last_q_max - newZp);

        float input_min;
        float input_max;
        getFakeQuantParams(inputType, levels, input_min, input_max);

        VPUIP::PwlQuantReqs* newQuantParams =
                new PwlQuantReqs({{input_min, input_max, inputType.getScale(), inputType.getZeroPoint(), 0},
                                  {fl_min, fl_max, newScale, newZp, 0}});
        return newQuantParams;
    } else
        return nullptr;
}
