//
// Copyright 2020 Intel Corporation.
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

#include <ie_blob.h>
#include <gtest/gtest.h>


namespace Comparators {
    void compareTopClasses(
            const InferenceEngine::Blob::Ptr &resultBlob, const InferenceEngine::Blob::Ptr &refBlob,
            size_t maxClasses, const bool &ordered = true);

    void compareTopClassesUnordered(
            const InferenceEngine::Blob::Ptr &resultBlob, const InferenceEngine::Blob::Ptr &refBlob,
            size_t maxClasses
    );

    template<typename blobData_t>
    std::vector<size_t>
    yieldTopClasses(const InferenceEngine::Blob::Ptr &resultBlob, size_t maxClasses);

    //------------------------------------------------------------------------------
    inline void compareTopClasses(
            const InferenceEngine::Blob::Ptr &resultBlob, const InferenceEngine::Blob::Ptr &refBlob,
            size_t maxClasses, const bool &ordered) {
        std::vector<size_t> outTopClasses, refTopClasses;
        switch (resultBlob->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::U8:
                outTopClasses = yieldTopClasses<uint8_t>(resultBlob, maxClasses);
                refTopClasses = yieldTopClasses<uint8_t>(refBlob, maxClasses);
                break;
            case InferenceEngine::Precision::FP32:
                outTopClasses = yieldTopClasses<float>(resultBlob, maxClasses);
                refTopClasses = yieldTopClasses<float>(refBlob, maxClasses);
                break;
            default:
                throw std::runtime_error("compareTopClasses: only U8 and FP32 are supported");
        }
        std::ostringstream logStream;
        logStream << "out: ";
        for (size_t classId = 0; classId < maxClasses; classId++) {
            logStream << outTopClasses[classId] << " ";
        }

        logStream << std::endl << "ref: ";
        for (size_t classId = 0; classId < maxClasses; classId++) {
            logStream << refTopClasses[classId] << " ";
        }
        if (ordered) {
            EXPECT_TRUE(std::equal(outTopClasses.begin(), outTopClasses.end(), refTopClasses.begin()))
                                << logStream.str();
        } else {
            EXPECT_TRUE(std::is_permutation(outTopClasses.begin(), outTopClasses.end(),
                                            refTopClasses.begin())) << logStream.str();
        }
    }

    inline void compareTopClassesUnordered(const InferenceEngine::Blob::Ptr &resultBlob,
                                           const InferenceEngine::Blob::Ptr &refBlob,
                                           size_t maxClasses) {
        compareTopClasses(resultBlob, refBlob, maxClasses, /*ordered*/false);
    }

    template<typename blobData_t>
    std::vector<size_t> yieldTopClasses(const InferenceEngine::Blob::Ptr &resultBlob, size_t maxClasses) {
        const blobData_t* unsortedRawData = resultBlob->cbuffer().as<const blobData_t*>();
        // map key is a byte from raw data (quantized probability)
        // map value is the index of that byte (class id)
        std::multimap<blobData_t, size_t> sortedClassMap;
        for (size_t classIndex = 0; classIndex < resultBlob->size(); classIndex++) {
            blobData_t classProbability = unsortedRawData[classIndex];
            std::pair<blobData_t, size_t> mapItem(classProbability, classIndex);
            sortedClassMap.insert(mapItem);
        }

        std::vector<size_t> topClasses;
        for (size_t classCounter = 0; classCounter < maxClasses; classCounter++) {
            typename std::multimap<blobData_t, size_t>::reverse_iterator classIter = sortedClassMap.rbegin();
            std::advance(classIter, classCounter);
            topClasses.push_back(classIter->second);
        }

        return topClasses;
    }

}
