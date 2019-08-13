#include <gtest/gtest.h>
#include <regression_tests.hpp>
#include <inference_engine/precision_utils.h>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/private_plugin_config.hpp>

#include <ie_icnn_network_stats.hpp>
#include <cnn_network_int8_normalizer.hpp>
#include <ie_util_internal.hpp>
#include <ie_compound_blob.h>

#include <vpu_layers_tests.hpp>

#include "vpusmm.h"

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

using namespace ::testing;
using namespace InferenceEngine;
using namespace Regression::Matchers;
using namespace InferenceEngine::details;

enum preprocessingType {
    PT_RESIZE, PT_NV12
};

class VpuPreprocessingTestsWithParam : public vpuLayersTests,
                                       public testing::WithParamInterface< preprocessingType > {
};

class VPUAllocator {
public:
    VPUAllocator() {};
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

Blob::Ptr fromNV12File(const std::string &filePath,
                       size_t imageWidth,
                       size_t imageHeight,
                       VPUAllocator &allocator) {
    std::ifstream fileReader(filePath, std::ios_base::ate | std::ios_base::binary);
    if (!fileReader.good()) {
        throw std::runtime_error("fromNV12File: failed to open file " + filePath);
    }

    const size_t expectedSize = imageWidth * (imageHeight * 3 / 2);
    const size_t fileSize = fileReader.tellg();
    if (fileSize < expectedSize) {
        throw std::runtime_error("fromNV12File: size of " + filePath + " is less than expected");
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

static std::string composePreprocInputPath(preprocessingType preprocType) {
    std::string baseName = ModelsPath() + "/KMB_models/BLOBS/mobilenet/";
    switch (preprocType) {
    case PT_RESIZE:
        baseName += "input-227x227.dat";
        break;
    case PT_NV12:
        baseName += "input-nv12.dat";
        break;
    }
    return baseName;
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

static void setPreprocForInputBlob(const std::string &inputName,
                                   const TensorDesc &inputTensor,
                                   const std::string &inputFilePath,
                                   InferenceEngine::InferRequest &inferRequest,
                                   VPUAllocator &allocator,
                                   preprocessingType preprocType) {
    Blob::Ptr inputBlob;
    switch (preprocType) {
    case PT_RESIZE: {
            uint8_t *imageData = reinterpret_cast<uint8_t*>(allocator.allocate(3 * 227 * 227));
            InferenceEngine::TensorDesc preprocTensor(
                inputTensor.getPrecision(),
                {1, 3, 227, 227},
                inputTensor.getLayout());
            inputBlob = make_shared_blob<uint8_t>(preprocTensor, imageData);
            ASSERT_TRUE(fromBinaryFile(inputFilePath, inputBlob));
        }
        break;
    case PT_NV12:
        const InferenceEngine::Layout inputLayout = inputTensor.getLayout();
        const InferenceEngine::SizeVector dims = inputTensor.getDims();
        const size_t expectedWidth = dims.at(3);
        const size_t expectedHeight = dims.at(2);
        ASSERT_NO_THROW(inputBlob = fromNV12File(inputFilePath, expectedWidth, expectedHeight, allocator));
        break;
    }
    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, inputBlob));
}

Blob::Ptr dequantize(float begin, float end, const Blob::Ptr &quantBlob, VPUAllocator &allocator) {
    const int QUANT_LEVELS = 256;
    float step = (begin - end)/QUANT_LEVELS;
    const TensorDesc quantTensor = quantBlob->getTensorDesc();
    const TensorDesc outTensor = TensorDesc(
        InferenceEngine::Precision::FP32,
        quantTensor.getDims(),
        quantTensor.getLayout());
    const uint8_t *quantRaw = quantBlob->cbuffer().as<const uint8_t *>();
    float *outRaw = reinterpret_cast<float *>(allocator.allocate(quantBlob->byteSize() * sizeof(float)));

    for (size_t pos = 0; pos < quantBlob->byteSize(); pos++) {
        outRaw[pos] = begin + quantRaw[pos] * step;
    }
    Blob::Ptr outputBlob = make_shared_blob<float>(outTensor, outRaw);
    return outputBlob;
}

TEST_P(VpuPreprocessingTestsWithParam, DISABLED_importWithPreprocessing) {  // To be run in manual mode when device is available
    preprocessingType preprocType = GetParam();
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet/mobilenet.blob";

    VPUAllocator kmbAllocator;

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, "KMB", {}));

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    for (auto & item : inputInfo) {
        InputInfo* mutableItem = const_cast<InputInfo*>(item.second.get());
        setPreprocAlgorithm(mutableItem, preprocType);
    }

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    std::string inputFilePath = composePreprocInputPath(preprocType);

    for (auto & item : inputInfo) {
        std::string inputName = item.first;
        InferenceEngine::TensorDesc inputTensor = item.second->getTensorDesc();
        setPreprocForInputBlob(inputName, inputTensor, inputFilePath, inferRequest, kmbAllocator, preprocType);
    }

    ASSERT_NO_THROW(inferRequest.Infer());

    ConstOutputsDataMap outputInfo;
    ASSERT_NO_THROW(outputInfo = importedNetwork.GetOutputsInfo());

    std::string referenceOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet/output.dat";
    for (auto & item : outputInfo) {
        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(item.first.c_str()));

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();

        uint8_t* outputRefData = reinterpret_cast<uint8_t*>(kmbAllocator.allocate(outputBlob->byteSize()));
        Blob::Ptr referenceOutputBlob = make_shared_blob<uint8_t>(outputBlobTensorDesc, outputRefData);
        ASSERT_TRUE(fromBinaryFile(referenceOutputFilePath, referenceOutputBlob));

        // Refer to mobilenet tflite to find these constants
        // https://github.com/movidius/migNetworkZoo
        const float MOBILE_NET_QUANT_START = -31.364717483520508;
        const float MOBILE_NET_QUANT_END = 2.2403368949890137;
        Blob::Ptr dequantRef = dequantize(MOBILE_NET_QUANT_START, MOBILE_NET_QUANT_END, referenceOutputBlob, kmbAllocator);
        Blob::Ptr dequantOut = dequantize(MOBILE_NET_QUANT_START, MOBILE_NET_QUANT_END, outputBlob, kmbAllocator);
        Compare(dequantRef, dequantOut, 100.f);
    }
}

const static std::vector<preprocessingType> preprocTypes = {
    PT_RESIZE, PT_NV12
};

INSTANTIATE_TEST_CASE_P(preprocessing, VpuPreprocessingTestsWithParam,
    ::testing::ValuesIn(preprocTypes)
);

