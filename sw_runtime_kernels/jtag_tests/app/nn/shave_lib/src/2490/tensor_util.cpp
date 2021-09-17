/*
* {% copyright %}
*/
#include "tensor_util.h"
#include <limits.h>
#include <nn_log.h>

namespace nn {
namespace shave_lib {

uint32_t getMiddleStride(const TensorRef *data) {
    uint32_t stride = 0;

    if (data != nullptr) {
        switch (data->ndOrder) {
        case ND_CHW:
        case ND_NCHW: {
            // case ND_XYZ: // TODO: fix remaining formats
            stride = data->strideH();
            break;
        }
            // case ND_YZX:
            // case ND_XZY:
            //    stride = data->dimZStride;
            //    break;

        case ND_HWC:
        case ND_NHWC: {
            // case ND_ZXY:
            stride = data->strideW();
            break;
        }
        default:
            // supresses build warnings
            nnLog(MVLOG_FATAL, "Unhandled stride order!");
            stride = UINT_MAX ;
            break;
        }
    }

    return stride;
}

uint32_t getMajorDim(const TensorRef *data) {
    uint32_t dim = 0;

    if (data != nullptr) {
        switch (data->ndOrder) {
        case ND_HWC:
        case ND_NHWC: {
            // case ND_YZX:    // TODO: fix remaining formats
            dim = data->dimH();
            break;
        }
        // case ND_XYZ:
        // case ND_XZY:
        //    dim = data->dimX;
        //    break;

        // case ND_ZXY:
        case ND_CHW:
        case ND_NCHW: {
            dim = data->dimC();
            break;
        }
        default: {
            // supresses build warnings
            nnLog(MVLOG_FATAL, "Unhandled stride order!");
            dim = UINT_MAX;
            break;
        }
        }
    }

    return dim;
}

uint32_t getMajorStride(const TensorRef *data) {
    uint32_t stride = 0;

    if (data != nullptr) {
        switch (data->ndOrder) {
        case ND_HWC:
        case ND_NHWC: {
            // case ND_YZX:    // TODO: fix remaining formats
            stride = data->strideH();
            break;
        }
        // case ND_XYZ:
        // case ND_XZY:
        //    stride = data->dimXStride;
        //    break;

        // case ND_ZXY:
        case ND_CHW:
        case ND_NCHW: {
            stride = data->strideC();
            break;
        }
        default: {
            // supresses build warnings
            nnLog(MVLOG_FATAL, "Unhandled stride order!");
            stride = UINT_MAX;
            break;
        }
        }
    }
    return stride;
}

TensorSplitter::TensorSplitter(uint32_t numLines, uint32_t numShaves, uint32_t bytesPerLine_)
    : bytesPerLine(bytesPerLine_) {
    numLines /= bytesPerLine;
    baseDivision = numLines / numShaves;
    remainder = numLines % numShaves;
    currentLine = 0;
    lastNumLines = 0;
}

uint32_t TensorSplitter::getNumLines() {
    uint32_t numLines = baseDivision + (remainder == 0 ? 0 : 1);

    if (remainder > 0)
        remainder--;

    currentLine += lastNumLines;
    lastNumLines = numLines;

    return numLines;
}

uint32_t TensorSplitter::getNumLinesInBytes() { return getNumLines() * bytesPerLine; }

uint32_t TensorSplitter::getCurrentLine() { return currentLine; }

uint32_t TensorSplitter::getCurrentOffsetInBytes() { return currentLine * bytesPerLine; }

} // namespace shave_lib
} // namespace nn
