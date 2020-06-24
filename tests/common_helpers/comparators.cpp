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

#include "comparators.h"

#include <cmath>

void Comparators::compareBoundingBoxes(std::vector<utils::YoloBBox>& actualOutput,
    std::vector<utils::YoloBBox>& refOutput, int imgWidth, int imgHeight, float boxTolerance, float probTolerance) {
    std::cout << "Ref Top:" << std::endl;
    for (size_t i = 0; i < refOutput.size(); ++i) {
        const auto& bb = refOutput[i];
        std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right << " "
                  << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
    }

    std::cout << "Actual top:" << std::endl;
    for (size_t i = 0; i < actualOutput.size(); ++i) {
        const auto& bb = actualOutput[i];
        std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right << " "
                  << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
    }

    for (const auto& refBB : refOutput) {
        bool found = false;

        float maxBoxError = 0.0f;
        float maxProbError = 0.0f;

        for (const auto& actualBB : actualOutput) {
            if (actualBB.idx != refBB.idx) {
                continue;
            }

            const utils::YoloBox actualBox {actualBB.left / imgWidth, actualBB.top / imgHeight,
                (actualBB.right - actualBB.left) / imgWidth, (actualBB.bottom - actualBB.top) / imgHeight};
            const utils::YoloBox refBox {refBB.left / imgWidth, refBB.top / imgHeight,
                (refBB.right - refBB.left) / imgWidth, (refBB.bottom - refBB.top) / imgHeight};

            const auto boxIntersection = boxIntersectionOverUnion(actualBox, refBox);
            const auto boxError = 1.0f - boxIntersection;
            maxBoxError = std::max(maxBoxError, boxError);

            const auto probError = std::fabs(actualBB.prob - refBB.prob);
            maxProbError = std::max(maxProbError, probError);

            if (boxError > boxTolerance) {
                continue;
            }

            if (probError > probTolerance) {
                continue;
            }

            found = true;
            break;
        }

        EXPECT_TRUE(found) << "maxBoxError=" << maxBoxError << " "
                           << "maxProbError=" << maxProbError;
    }
}
