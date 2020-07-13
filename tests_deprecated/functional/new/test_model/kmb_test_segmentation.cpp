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

#include "kmb_test_base.hpp"

#include <iostream>

/*
 *  Semantic segmentation
 */
static std::vector<long> fp32toNearestLong(const Blob::Ptr& ieBlob) {
    std::vector<long> nearestLongCollection;
    const float* blobRawPtr = ieBlob->cbuffer().as<const float*>();
    for (size_t pos = 0; pos < ieBlob->size(); pos++) {
        long nearestLong = std::lround(blobRawPtr[pos]);
        nearestLongCollection.push_back(nearestLong);
    }

    return nearestLongCollection;
}

static std::vector<long> int32toLong(const Blob::Ptr& ieBlob) {
    std::vector<long> nearestLongCollection;
    const int32_t* blobRawPtr = ieBlob->cbuffer().as<const int32_t*>();
    for (size_t pos = 0; pos < ieBlob->size(); pos++) {
        nearestLongCollection.push_back(blobRawPtr[pos]);
    }

    return nearestLongCollection;
}

/*
 * Calculates intersections and unions by definition for each label in range.
 * Naive implementation. Extremely slow. Performs Lx2xHxW comparisons, where
 * L - number of labels in the range
 * H - height of the segmented image
 * W - width of the segmented image
 */
static float calculateMeanIntersectionOverUnion(
        const std::vector<long>& vpuOut,
        const std::vector<long>& cpuOut,
        const long& firstLabel,
        const long& lastLabel) {
    float totalIoU = 0.f;
    size_t nonZeroUnions = 0;
    for (long labelId = firstLabel; labelId < lastLabel; labelId++) {
        size_t intersectionSize = 0;
        size_t unionSize = 0;
        for (size_t pos = 0; pos < vpuOut.size() && pos < cpuOut.size(); pos++) {
            bool vpuLabelMatches = (vpuOut.at(pos) == labelId);
            bool cpuLabelMatches = (cpuOut.at(pos) == labelId);
            // intersection is when both values match the label
            if (vpuLabelMatches && cpuLabelMatches) {
                intersectionSize++;
            }

            // union is when at least one of the values matches the label
            if (vpuLabelMatches || cpuLabelMatches) {
                unionSize++;
            }
        }

        if (unionSize > 0) {
            // cast to floating point to avoid integer division
            float intersectionCardinality = intersectionSize;
            float unionCardinality = unionSize;
            float classIoU = intersectionCardinality / unionCardinality;
            std::cout << "Label: " << labelId << " IoU: " << classIoU << std::endl;
            nonZeroUnions++;
            totalIoU += classIoU;
        }
    }

    float meanIoU = totalIoU / nonZeroUnions;
    return meanIoU;
}

void KmbSegmentationNetworkTest::runTest(
        const TestNetworkDesc& netDesc,
        const TestImageDesc& image,
        const long& firstLabel,
        const long& lastLabel,
        const float& meanIntersectionOverUnionTolerance) {
    const auto check = [=](const Blob::Ptr& actualBlob, const Blob::Ptr& refBlob, const TensorDesc&) {
        // FIXME VPU compiler overrides any output precision to FP32 when asked
        // CPU doesn't override I32 output precision during compilation
        std::vector<long> vpuOut = fp32toNearestLong(toFP32(actualBlob));
        std::vector<long> cpuOut = int32toLong(refBlob);
        ASSERT_EQ(vpuOut.size(), cpuOut.size())
            << "vpuOut.size: " << vpuOut.size() << " "
            << "cpuOut.size: " << cpuOut.size();

        float meanIoU = calculateMeanIntersectionOverUnion(vpuOut, cpuOut, firstLabel, lastLabel);
        EXPECT_LE(meanIntersectionOverUnionTolerance, meanIoU)
            << "meanIoU: " << meanIoU << " "
            << "meanIoUTolerance: " << meanIntersectionOverUnionTolerance;
    };

    KmbNetworkTestBase::runTest(netDesc, image, check);
}
