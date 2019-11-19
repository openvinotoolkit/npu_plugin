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


#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <tuple>
#include <list>

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <samples/args_helper.hpp>

#include "detection_sample_yolov2tiny.h"
#include "region_yolov2tiny.h"

#include <ie_icnn_network_stats.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <ie_util_internal.hpp>
#include <ie_compound_blob.h>

#include "vpusmm.h"

#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace InferenceEngine;
ConsoleErrorListener error_listener;

enum preprocessingType {
    PT_RESIZE, PT_NV12
};

class VPUAllocator {
public:
    VPUAllocator() {}
    virtual ~VPUAllocator();
    void* allocate(size_t requestedSize);
private:
    std::list< std::tuple<int, void*, size_t> > _memChunks;
    static int _pageSize;
};

int VPUAllocator::_pageSize = getpagesize();

static uint32_t calculateRequiredSize(uint32_t blobSize, int pageSize) {
    uint32_t blobSizeRem = blobSize % pageSize;
    uint32_t requiredSize = (blobSize / pageSize) * pageSize;
    if (blobSizeRem) {
        requiredSize += pageSize;
    }
    return requiredSize;
}

void* VPUAllocator::allocate(size_t requestedSize) {
    const uint32_t requiredBlobSize = calculateRequiredSize(requestedSize, _pageSize);
    int fileDesc = vpusmm_alloc_dmabuf(requiredBlobSize, VPUSMMTYPE_COHERENT);
    if (fileDesc < 0) {
        throw std::runtime_error("VPUAllocator::allocate: vpusmm_alloc_dmabuf failed");
    }

    unsigned long physAddr = vpusmm_import_dmabuf(fileDesc, VPU_DEFAULT);
    if (physAddr == 0) {
        throw std::runtime_error("VPUAllocator::allocate: vpusmm_import_dmabuf failed");
    }

    void* virtAddr = mmap(0, requiredBlobSize, PROT_READ|PROT_WRITE, MAP_SHARED, fileDesc, 0);
    if (virtAddr == MAP_FAILED) {
        throw std::runtime_error("VPUAllocator::allocate: mmap failed");
    }
    std::tuple<int, void*, size_t> memChunk(fileDesc, virtAddr, requiredBlobSize);
    _memChunks.push_back(memChunk);

    return virtAddr;
}

VPUAllocator::~VPUAllocator() {
    for (const std::tuple<int, void*, size_t> & chunk : _memChunks) {
        int fileDesc = std::get<0>(chunk);
        void* virtAddr = std::get<1>(chunk);
        size_t allocatedSize = std::get<2>(chunk);
        vpusmm_unimport_dmabuf(fileDesc);
        munmap(virtAddr, allocatedSize);
        close(fileDesc);
    }
}

static void setPreprocAlgorithm(InputInfo* mutableItem, preprocessingType preprocType) {
    switch (preprocType) {
    case PT_RESIZE:
        mutableItem->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        break;
    case PT_NV12:
        mutableItem->getPreProcess().setColorFormat(ColorFormat::NV12);
        break;
    }
}

static Blob::Ptr fromNV12File(const std::string &filePath,
                       size_t imageWidth,
                       size_t imageHeight,
                       VPUAllocator &allocator) {
    std::ifstream fileReader(filePath, std::ios_base::ate | std::ios_base::binary);
    if (!fileReader.good()) {
        throw std::runtime_error("fromNV12File: failed to open file " + filePath);
    }

    const size_t expectedSize = (imageWidth * imageHeight * 3 / 2);
    const size_t fileSize = fileReader.tellg();
    if (fileSize != expectedSize) {
        throw std::runtime_error("fromNV12File: size of " + filePath + " is not expected");
    }
    fileReader.seekg(0);

    uint8_t *imageData = reinterpret_cast<uint8_t *>(allocator.allocate(fileSize));
    if (!imageData) {
        throw std::runtime_error("fromNV12File: failed to allocate memory");
    }

    fileReader.read(reinterpret_cast<char *>(imageData), fileSize);
    fileReader.close();

    InferenceEngine::TensorDesc planeY(InferenceEngine::Precision::U8,
        {1, 1, imageHeight, imageWidth}, InferenceEngine::Layout::NHWC);
    InferenceEngine::TensorDesc planeUV(InferenceEngine::Precision::U8,
        {1, 2, imageHeight / 2, imageWidth / 2}, InferenceEngine::Layout::NHWC);
    const size_t offset = imageHeight * imageWidth;

    Blob::Ptr blobY = make_shared_blob<uint8_t>(planeY, imageData);
    Blob::Ptr blobUV = make_shared_blob<uint8_t>(planeUV, imageData + offset);

    Blob::Ptr nv12Blob = make_shared_blob<NV12Blob>(blobY, blobUV);
    return nv12Blob;
}

static void setPreprocForInputBlob(const std::string &inputName,
                                   const TensorDesc &inputTensor,
                                   const std::string &inputFilePath,
                                   InferenceEngine::InferRequest &inferRequest,
                                   VPUAllocator &allocator) {
    Blob::Ptr inputBlobNV12;
    const InferenceEngine::SizeVector dims = inputTensor.getDims();
    const size_t expectedWidth = FLAGS_iw;
    const size_t expectedHeight = FLAGS_ih;
    inputBlobNV12 = fromNV12File(inputFilePath, expectedWidth, expectedHeight, allocator);

    inferRequest.SetBlob(inputName, inputBlobNV12);
}

Blob::Ptr deQuantize(const Blob::Ptr &quantBlob, float scale, uint8_t zeroPoint) {
    const TensorDesc quantTensor = quantBlob->getTensorDesc();
    const TensorDesc outTensor = TensorDesc(
        InferenceEngine::Precision::FP32,
        quantTensor.getDims(),
        quantTensor.getLayout());
    const uint8_t *quantRaw = quantBlob->cbuffer().as<const uint8_t *>();

    std::vector<size_t> dims = quantTensor.getDims();

    Blob::Ptr outputBlob = make_shared_blob<float>(outTensor);
    outputBlob->allocate();
    float *outRaw = outputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    for (size_t pos = 0; pos < quantBlob->byteSize(); pos++) {
        outRaw[pos] = (quantRaw[pos] - zeroPoint) * scale;
    }

    return outputBlob;
}

Blob::Ptr yoloLayer_yolov2tiny(const Blob::Ptr &lastBlob, int inputHeight, int inputWidth) {
    const TensorDesc quantTensor = lastBlob->getTensorDesc();
    const TensorDesc outTensor = TensorDesc(InferenceEngine::Precision::FP32,
        {1, 1, 13*13*20*5, 7},
        lastBlob->getTensorDesc().getLayout());
    Blob::Ptr outputBlob = make_shared_blob<float>(outTensor);
    outputBlob->allocate();

    const float *inputRawData = lastBlob->cbuffer().as<const float *>();
    float *outputRawData = outputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();

    int shape[]={13, 13, 5, 25};
    int strides[]={13*128, 128, 25, 1};
    postprocess::yolov2(inputRawData, shape, strides,
        0.4f, 0.45f, 20, 416, 416, outputRawData);

    return outputBlob;
}

//===========================================================================

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_iw == 0) {
        throw std::logic_error("Parameter -iw is not set");
    }

    if (FLAGS_ih == 0) {
        throw std::logic_error("Parameter -ih is not set");
    }

    return true;
}

bool readBinaryFile(std::string input_binary, std::string& data) {
    std::ifstream in(input_binary, std::ios_base::binary | std::ios_base::ate);

    size_t sizeFile = in.tellg();
    in.seekg(0, std::ios_base::beg);
    data.resize(sizeFile);
    bool status = false;
    if (in.good()) {
        in.read(&data.front(), sizeFile);
        status = true;
    }
    return status;
}

std::vector<std::string> readLabelsFromFile(const std::string& labelFileName) {
    std::vector<std::string> labels;

    std::ifstream inputFile;
    inputFile.open(labelFileName, std::ios::in);
    if (inputFile.is_open()) {
        std::string strLine;
        while (std::getline(inputFile, strLine)) {
            trim(strLine);
            labels.push_back(strLine);
        }
    }
    return labels;
}

/**
* @brief The entry point the Inference Engine sample application
* @file detection_sample/main.cpp
* @example detection_sample/main.cpp
*/
int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;
        VPUAllocator kmbAllocator;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::string imageFileName = FLAGS_i;

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Creating Inference Engine" << slog::endl;
        Core ie;

        if (FLAGS_p_msg) {
            ie.SetLogCallback(error_listener);
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read blob Generated by MCM Compiler ----------------------------------
        std::string binFileName = FLAGS_m;
        slog::info << "Loading blob:\t" << binFileName << slog::endl;

        ExecutableNetwork importedNetwork = ie.ImportNetwork(binFileName, "KMB", {});
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

        for (auto & item : inputInfo) {
            InputInfo* mutableItem = const_cast<InputInfo*>(item.second.get());
            setPreprocAlgorithm(mutableItem, PT_NV12);
            setPreprocAlgorithm(mutableItem, PT_RESIZE);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Create infer request -------------------------------------------------
        InferenceEngine::InferRequest inferRequest = importedNetwork.CreateInferRequest();
        slog::info << "CreateInferRequest completed successfully" << slog::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Prepare input --------------------------------------------------------
        /** Iterate over all the input blobs **/
        std::string firstInputName = inputInfo.begin()->first;

        /** Creating input blob **/
        Blob::Ptr inputBlob = inferRequest.GetBlob(firstInputName.c_str());
        if (!inputBlob) {
            throw std::logic_error("Cannot get input blob from inferRequest");
        }

        for (auto & item : inputInfo) {
            std::string inputName = item.first;
            InferenceEngine::TensorDesc inputTensor = item.second->getTensorDesc();
            setPreprocForInputBlob(inputName, inputTensor, imageFileName, inferRequest, kmbAllocator);
        }

        inferRequest.Infer();
        slog::info << "inferRequest completed successfully" << slog::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        ConstOutputsDataMap outputInfo = importedNetwork.GetOutputsInfo();
        if (outputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 output");

        std::string firstOutputName = outputInfo.begin()->first;

        Blob::Ptr outputBlob = inferRequest.GetBlob(firstOutputName.c_str());
        if (!outputBlob) {
            throw std::logic_error("Cannot get output blob from inferRequest");
        }

        // de-Quantization
        uint8_t zeroPoint = static_cast<uint8_t>(FLAGS_z);
        float scale = static_cast<float>(FLAGS_s);

        // Real data layer
        Blob::Ptr dequantOut = deQuantize(outputBlob, scale, zeroPoint);

        // Region YOLO layer
        int inputHeight = inputBlob->getTensorDesc().getDims()[2];
        int inputWidth = inputBlob->getTensorDesc().getDims()[3];
        Blob::Ptr detectResult = yoloLayer_yolov2tiny(dequantOut, inputHeight, inputWidth);

        // Print result.
        size_t N = detectResult->getTensorDesc().getDims()[2];
        if (detectResult->getTensorDesc().getDims()[3] != 7) {
            throw std::logic_error("Output item should have 7 as a last dimension");
        }
        const float *rawData = detectResult->cbuffer().as<const float *>();
        // imageid,labelid,confidence,x0,y0,x1,y1
        for (size_t i = 0; i < N; i++) {
            if (rawData[i*7 + 2] > 0.001) {
                slog::info << "label = " << postprocess::YOLOV2_TINY_LABELS.at(rawData[i*7 + 1]) << slog::endl;
                slog::info << "confidence = " << rawData[i*7 + 2] << slog::endl;
                slog::info << "x0,y0,x1,y1 = " << rawData[i*7 + 3] << ", "
                    << rawData[i*7 + 4] << ", "
                    << rawData[i*7 + 5] << ", "
                    << rawData[i*7 + 6] << slog::endl;
            }
        }

        std::fstream outFile;
        outFile.open("/output.dat", std::ios::in | std::ios::out | std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(outputBlob->buffer(), outputBlob->size());
        }
        outFile.close();
    }
    catch (const std::exception& error) {
        slog::err << "" << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
