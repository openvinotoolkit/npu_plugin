//
// Copyright 2019 Intel Corporation.
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

#include "file_reader.h"

#include <precision_utils.h>
#include <fstream>

namespace vpu {

namespace KmbPlugin {

namespace utils {

void fromBinaryFile(std::string input_binary, InferenceEngine::Blob::Ptr blob) {
    std::ifstream in(input_binary, std::ios_base::binary | std::ios_base::ate);

    size_t sizeFile = in.tellg();
    in.seekg(0, std::ios_base::beg);
    size_t count = blob->size();
    if (in.good()) {
        if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP16) {
            InferenceEngine::ie_fp16 *blobRawDataFP16 = blob->buffer().as<InferenceEngine::ie_fp16 *>();
            if (sizeFile == count * sizeof(float)) {
                for (size_t i = 0; i < count; i++) {
                    float tmp;
                    in.read(reinterpret_cast<char *>(&tmp), sizeof(float));
                    blobRawDataFP16[i] = InferenceEngine::PrecisionUtils::f32tof16(tmp);
                }
            } else if (sizeFile == count * sizeof(InferenceEngine::Precision::FP16)) {
                for (size_t i = 0; i < count; i++) {
                    InferenceEngine::ie_fp16 tmp;
                    in.read(reinterpret_cast<char *>(&tmp), sizeof(InferenceEngine::ie_fp16));
                    blobRawDataFP16[i] = tmp;
                }
            } else {
                THROW_IE_EXCEPTION << "File has invalid size!";
            }
        } else if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP32) {
            float *blobRawData = blob->buffer();
            if (sizeFile == count * sizeof(float)) {
                in.read(reinterpret_cast<char *>(blobRawData), count * sizeof(float));
            } else {
                THROW_IE_EXCEPTION << "File has invalid size!";
            }
        } else if (blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::U8) {
            char *blobRawData = blob->buffer().as<char *>();
            if (sizeFile == count * sizeof(char)) {
                in.read(blobRawData, count * sizeof(char));
            } else {
                THROW_IE_EXCEPTION << "File has invalid size!";
            }
        }
    } else {
        THROW_IE_EXCEPTION << "File is not good.";
    }
}

}  // namespace utils

}  // namespace KmbPlugin

}  // namespace vpu
