/*
* {% copyright %}
*/
#pragma once

#include "sw_tensor_ref.h"
#include <algorithm>
#include <type_traits>

namespace nn {
namespace shave_lib {

uint32_t getMiddleStride(const TensorRef *data);
uint32_t getMajorDim(const TensorRef *data);
uint32_t getMajorStride(const TensorRef *data);

// Split a tensor into set of lines for parallel processing
// Automatically add the remaining lines to the first batches
// when the dimension is not evenly divisible by the number of
// batches (shaves)
class TensorSplitter {
    public:
    TensorSplitter(uint32_t numLines, uint32_t numShaves, uint32_t bytesPerLine = 1);

    // Return the number of lines this shave should process
    uint32_t getNumLines();
    uint32_t getNumLinesInBytes();

    // Return the current line offset for this shave. Should be called after
    // getNumLines().
    // On call 0, returns 0
    // On call n, returns Sum0..n(getNumLines())
    // May be called multiple times for each invocation of getNumLines() and will return the same value
    uint32_t getCurrentLine();
    uint32_t getCurrentOffsetInBytes();

    private:
    uint32_t baseDivision;
    int32_t remainder;
    uint32_t currentLine;
    uint32_t lastNumLines;
    uint32_t bytesPerLine;
};

} // namespace shave_lib
} // namespace nn
