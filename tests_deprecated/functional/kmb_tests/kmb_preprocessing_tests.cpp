#if defined(__arm__) || defined(__aarch64__)

#include <fcntl.h>
#include <file_reader.h>
#include <gtest/gtest.h>
#include <ie_compound_blob.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vpusmm/vpusmm.h>

#include <condition_variable>
#include <mutex>
#include <regression_tests.hpp>
#include <test_model/kmb_test_utils.hpp>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu_layers_tests.hpp>

#include "test_model/kmb_test_base.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace Regression::Matchers;
using namespace InferenceEngine::details;

enum preprocessingType { PT_RESIZE, PT_NV12 };

class VpuPreprocessingTestsWithParam : public vpuLayersTests, public testing::WithParamInterface<preprocessingType> {};

static std::string composePreprocInputPath(preprocessingType preprocType) {
    std::string baseName = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/";
    switch (preprocType) {
    case PT_RESIZE:
        baseName += "input-227x227.bin";
        break;
    case PT_NV12:
        baseName += "input-228x228-nv12.bin";
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

static void setPreprocForInputBlob(const std::string& inputName, const TensorDesc& inputTensor,
    const std::string& inputFilePath, InferenceEngine::InferRequest& inferRequest,
    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator>& allocator, preprocessingType preprocType) {
    Blob::Ptr inputBlob;
    switch (preprocType) {
    case PT_RESIZE: {
        uint8_t* imageData = reinterpret_cast<uint8_t*>(allocator->allocate(3 * 227 * 227));
        InferenceEngine::TensorDesc preprocTensor(
            inputTensor.getPrecision(), {1, 3, 227, 227}, inputTensor.getLayout());
        inputBlob = make_shared_blob<uint8_t>(preprocTensor, imageData);
        ASSERT_NO_THROW(inputBlob = vpu::KmbPlugin::utils::fromBinaryFile(inputFilePath, preprocTensor));
    } break;
    case PT_NV12:
        const InferenceEngine::SizeVector dims = inputTensor.getDims();
        const size_t expectedWidth = dims.at(2);
        const size_t expectedHeight = dims.at(3);
        ASSERT_NO_THROW(
            inputBlob = vpu::KmbPlugin::utils::fromNV12File(inputFilePath, expectedWidth, expectedHeight, allocator));
        break;
    }
    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, inputBlob));
}

static void setNV12Preproc(const std::string& inputName, const std::string& inputFilePath,
    InferenceEngine::InferRequest& inferRequest, std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator>& allocator,
    size_t expectedWidth, size_t expectedHeight) {
    Blob::Ptr inputBlob;
    ASSERT_NO_THROW(
        inputBlob = vpu::KmbPlugin::utils::fromNV12File(inputFilePath, expectedWidth, expectedHeight, allocator));

    PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setResizeAlgorithm(RESIZE_BILINEAR);
    preprocInfo.setColorFormat(ColorFormat::NV12);

    ASSERT_NO_THROW(inferRequest.SetBlob(inputName, inputBlob, preprocInfo));
}

static void setRandomNV12(const std::string& inputName, InferenceEngine::InferRequest& inferRequest,
    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator>& allocator, size_t imageWidth, size_t imageHeight) {
    const size_t expectedSize = imageWidth * (imageHeight * 3 / 2);
    uint8_t* imageData = reinterpret_cast<uint8_t*>(allocator->allocate(expectedSize));
    std::random_device randDev;
    std::default_random_engine randEngine(randDev());
    std::uniform_int_distribution<uint8_t> uniformDist(0, 255);
    for (size_t byteCount = 0; byteCount < expectedSize; byteCount++) {
        imageData[byteCount] = uniformDist(randEngine);
    }
    InferenceEngine::TensorDesc planeY(
        InferenceEngine::Precision::U8, {1, 1, imageHeight, imageWidth}, InferenceEngine::Layout::NHWC);
    InferenceEngine::TensorDesc planeUV(
        InferenceEngine::Precision::U8, {1, 2, imageHeight / 2, imageWidth / 2}, InferenceEngine::Layout::NHWC);
    const size_t offset = imageHeight * imageWidth;

    InferenceEngine::Blob::Ptr blobY = InferenceEngine::make_shared_blob<uint8_t>(planeY, imageData);
    InferenceEngine::Blob::Ptr blobUV = InferenceEngine::make_shared_blob<uint8_t>(planeUV, imageData + offset);

    PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
    preprocInfo.setResizeAlgorithm(RESIZE_BILINEAR);
    preprocInfo.setColorFormat(ColorFormat::NV12);

    InferenceEngine::Blob::Ptr nv12Blob = InferenceEngine::make_shared_blob<InferenceEngine::NV12Blob>(blobY, blobUV);
    inferRequest.SetBlob(inputName, nv12Blob, preprocInfo);
}

Blob::Ptr dequantize(float begin, float end, const Blob::Ptr& quantBlob,
    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator>& allocator) {
    const int QUANT_LEVELS = 256;
    float step = (begin - end) / QUANT_LEVELS;
    const TensorDesc quantTensor = quantBlob->getTensorDesc();
    const TensorDesc outTensor =
        TensorDesc(InferenceEngine::Precision::FP32, quantTensor.getDims(), quantTensor.getLayout());
    const uint8_t* quantRaw = quantBlob->cbuffer().as<const uint8_t*>();
    float* outRaw = reinterpret_cast<float*>(allocator->allocate(quantBlob->byteSize() * sizeof(float)));

    for (size_t pos = 0; pos < quantBlob->byteSize(); pos++) {
        outRaw[pos] = begin + quantRaw[pos] * step;
    }
    Blob::Ptr outputBlob = make_shared_blob<float>(outTensor, outRaw);
    return outputBlob;
}

std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> buildAllocator(const char* allocatorType) {
    if (allocatorType == nullptr) {
        return std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
    }

    std::string allocTypeStr(allocatorType);
    if (allocTypeStr == "NATIVE") {
        return std::make_shared<vpu::KmbPlugin::utils::NativeAllocator>();
    } else if (allocTypeStr == "UDMA") {
        throw std::runtime_error("buildAllocator: UDMA is not supported");
    }

    // VPUSMM is default
    return std::make_shared<vpu::KmbPlugin::utils::VPUSMMAllocator>();
}

// [Track number: S#21513]
TEST_P(VpuPreprocessingTestsWithParam,
    DISABLED_importWithPreprocessing) {  // To be run in manual mode when device is available
    preprocessingType preprocType = GetParam();
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, deviceName, {}));

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    for (auto& item : inputInfo) {
        InputInfo* mutableItem = const_cast<InputInfo*>(item.second.get());
        setPreprocAlgorithm(mutableItem, preprocType);
    }

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    std::string inputFilePath = composePreprocInputPath(preprocType);

    for (auto& item : inputInfo) {
        std::string inputName = item.first;
        InferenceEngine::TensorDesc inputTensor = item.second->getTensorDesc();
        setPreprocForInputBlob(inputName, inputTensor, inputFilePath, inferRequest, kmbAllocator, preprocType);
    }

    ASSERT_NO_THROW(inferRequest.Infer());

    ConstOutputsDataMap outputInfo;
    ASSERT_NO_THROW(outputInfo = importedNetwork.GetOutputsInfo());

    std::string referenceOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output-228x228-nv12.bin";
    for (auto& item : outputInfo) {
        Blob::Ptr outputBlob;
        ASSERT_NO_THROW(outputBlob = inferRequest.GetBlob(item.first.c_str()));

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();

        uint8_t* outputRefData = reinterpret_cast<uint8_t*>(kmbAllocator->allocate(outputBlob->byteSize()));
        Blob::Ptr referenceOutputBlob = make_shared_blob<uint8_t>(outputBlobTensorDesc, outputRefData);
        ASSERT_NO_THROW(
            referenceOutputBlob = vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, outputBlobTensorDesc));

        const size_t NUMBER_OF_CLASSES = 5;
        ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceOutputBlob, NUMBER_OF_CLASSES));
    }
}

using VpuPreprocessingTests = vpuLayersTests;

// [Track number: S#27548]
TEST_F(VpuPreprocessingTests, preprocResizeAndCSC) {
#if !defined(__arm__) && !defined(__aarch64__)
    SKIP();
#else
    if (useSIPP()) {
        SKIP();
    }
#endif
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, deviceName, {}));

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    inputInfo = importedNetwork.GetInputsInfo();
    std::string input_name = inputInfo.begin()->first;

    std::string inputFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input-228x228-nv12.bin";

    setNV12Preproc(input_name, inputFilePath, inferRequest, kmbAllocator, 228, 228);

    ASSERT_NO_THROW(inferRequest.Infer());

    ConstOutputsDataMap outputInfo;
    ASSERT_NO_THROW(outputInfo = importedNetwork.GetOutputsInfo());

    std::string referenceOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output-228x228-nv12.bin";
    for (auto& item : outputInfo) {
        Blob::Ptr outputBlob = toFP32(inferRequest.GetBlob(item.first.c_str()));

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();

        Blob::Ptr fileContentBlob;
        ASSERT_NO_THROW(
            fileContentBlob = vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, outputBlobTensorDesc));

        Blob::Ptr referenceOutputBlob = fileContentBlob;

        const size_t NUMBER_OF_CLASSES = 5;
        ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceOutputBlob, NUMBER_OF_CLASSES));
    }
}

// [Track number: S#27548]
TEST_F(VpuPreprocessingTests, multiThreadPreprocResizeAndCSC) {
#if !defined(__arm__) && !defined(__aarch64__)
    SKIP();
#else
    if (useSIPP()) {
        SKIP();
    }
#endif
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, deviceName, {}));

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    InferenceEngine::InferRequest inferRequest2;
    ASSERT_NO_THROW(inferRequest2 = importedNetwork.CreateInferRequest());

    inputInfo = importedNetwork.GetInputsInfo();
    std::string input_name = inputInfo.begin()->first;

    std::string inputFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input-228x228-nv12.bin";

    setNV12Preproc(input_name, inputFilePath, inferRequest, kmbAllocator, 228, 228);
    setNV12Preproc(input_name, inputFilePath, inferRequest2, kmbAllocator, 228, 228);

    ASSERT_NO_THROW(inferRequest.StartAsync());
    ASSERT_NO_THROW(inferRequest2.StartAsync());

    const unsigned WAIT_TIMEOUT = 60000;
    inferRequest.Wait(WAIT_TIMEOUT);
    inferRequest2.Wait(WAIT_TIMEOUT);

    ConstOutputsDataMap outputInfo;
    ASSERT_NO_THROW(outputInfo = importedNetwork.GetOutputsInfo());

    std::string referenceOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output-228x228-nv12.bin";
    for (auto& item : outputInfo) {
        Blob::Ptr outputBlob = toFP32(inferRequest.GetBlob(item.first.c_str()));

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();

        Blob::Ptr fileContentBlob;
        ASSERT_NO_THROW(
            fileContentBlob = vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, outputBlobTensorDesc));

        Blob::Ptr referenceOutputBlob = fileContentBlob;

        const size_t NUMBER_OF_CLASSES = 5;
        ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceOutputBlob, NUMBER_OF_CLASSES));
    }
}

// [Track number: S#27548]
TEST_F(VpuPreprocessingTests, twoRequestsWithPreprocessing) {
#if !defined(__arm__) && !defined(__aarch64__)
    SKIP();
#else
    if (useSIPP()) {
        SKIP();
    }
#endif
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    Core ie;
    InferenceEngine::ExecutableNetwork importedNetwork;
    ASSERT_NO_THROW(importedNetwork = ie.ImportNetwork(modelFilePath, deviceName, {}));

    ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

    inputInfo = importedNetwork.GetInputsInfo();
    std::string input_name = inputInfo.begin()->first;

    std::string inputFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input-228x228-nv12.bin";

    setNV12Preproc(input_name, inputFilePath, inferRequest, kmbAllocator, 228, 228);

    const std::size_t MAX_ITERATIONS = 10;
    volatile std::size_t iterationCount = 0;
    std::condition_variable waitCompletion;

    auto onComplete = [&input_name, &kmbAllocator, &waitCompletion, &iterationCount, &inferRequest, &inputFilePath](
                          void) -> void {
        iterationCount++;
        if (iterationCount < MAX_ITERATIONS) {
            setNV12Preproc(input_name, inputFilePath, inferRequest, kmbAllocator, 228, 228);
            ASSERT_NO_THROW(inferRequest.StartAsync());
        } else {
            waitCompletion.notify_one();
        }
    };

    inferRequest.SetCompletionCallback(onComplete);

    ASSERT_NO_THROW(inferRequest.StartAsync());

    std::mutex execGuard;
    std::unique_lock<std::mutex> execLocker(execGuard);
    waitCompletion.wait(execLocker, [&] {
        return iterationCount == MAX_ITERATIONS;
    });

    ConstOutputsDataMap outputInfo;
    ASSERT_NO_THROW(outputInfo = importedNetwork.GetOutputsInfo());

    std::string referenceOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/output-228x228-nv12.bin";
    for (auto& item : outputInfo) {
        Blob::Ptr outputBlob = toFP32(inferRequest.GetBlob(item.first.c_str()));

        TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();

        Blob::Ptr fileContentBlob;
        ASSERT_NO_THROW(
            fileContentBlob = vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, outputBlobTensorDesc));

        Blob::Ptr referenceOutputBlob = toFP32(fileContentBlob);

        const size_t NUMBER_OF_CLASSES = 5;
        ASSERT_NO_THROW(compareTopClasses(toFP32(outputBlob), referenceOutputBlob, NUMBER_OF_CLASSES));
    }
}
class VpuPreprocessingWithTwoNetworksTests :
    public VpuPreprocessingTests,
    public testing::WithParamInterface<
        std::tuple<const char*, std::size_t, std::size_t, const char*, std::size_t, std::size_t>> {};

TEST_P(VpuPreprocessingWithTwoNetworksTests, inference) {
    InferenceEngine::ExecutableNetwork network1;
    std::string network1Path = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";
    ASSERT_NO_THROW(network1 = core->ImportNetwork(network1Path, deviceName, {}));

    std::string network2Path = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/tiny-yolo-v2.blob";
    InferenceEngine::ExecutableNetwork network2;
    ASSERT_NO_THROW(network2 = core->ImportNetwork(network2Path, deviceName, {}));

    std::cout << "Created networks\n";

    ASSERT_EQ(1, network1.GetInputsInfo().size());
    ASSERT_EQ(1, network2.GetInputsInfo().size());
    std::cout << "Input info is OK\n";

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    InferenceEngine::InferRequest::Ptr network1InferReqPtr;
    network1InferReqPtr = network1.CreateInferRequestPtr();

    ConstInputsDataMap inputInfo1 = network1.GetInputsInfo();
    ConstInputsDataMap inputInfo2 = network2.GetInputsInfo();

    std::string input1_name = inputInfo1.begin()->first;
    auto param = GetParam();
    auto input1Path = ModelsPath() + get<0>(param);
    auto input1Width = get<1>(param);
    auto input1Height = get<2>(param);
    std::cout << "Set input for network1 with width = " << input1Width << ", height = " << input1Height << std::endl;
    setNV12Preproc(input1_name, input1Path, *network1InferReqPtr, kmbAllocator, input1Width, input1Height);

    InferenceEngine::InferRequest::Ptr network2InferReqPtr;
    network2InferReqPtr = network2.CreateInferRequestPtr();

    std::string input2_name = inputInfo2.begin()->first;
    auto input2Path = ModelsPath() + get<3>(param);
    auto input2Width = get<4>(param);
    auto input2Height = get<5>(param);
    std::cout << "Set input for network2 with width = " << input2Width << ", height = " << input2Height << std::endl;
    setNV12Preproc(input2_name, input2Path, *network2InferReqPtr, kmbAllocator, input2Width, input2Height);

    std::cout << "Created inference requests\n";

    ASSERT_EQ(1, network1.GetOutputsInfo().size());
    ASSERT_EQ(1, network2.GetOutputsInfo().size());
    std::cout << "Output info is OK\n";

    const auto iterationCount = 5;
    size_t curIterationNetwork1 = 0;
    size_t curIterationNet2 = 0;
    std::condition_variable condVar;

    network1InferReqPtr->SetCompletionCallback([&] {
        curIterationNetwork1++;
        std::cout << "Completed " << curIterationNetwork1 << " async request execution for network1\n";
        if (curIterationNetwork1 < static_cast<size_t>(iterationCount)) {
            Blob::Ptr outputBlob;
            std::string output1Name = network1.GetOutputsInfo().begin()->first;
            ASSERT_NO_THROW(outputBlob = network1InferReqPtr->GetBlob(output1Name));
            network1InferReqPtr->StartAsync();
        } else {
            condVar.notify_one();
        }
    });
    network2InferReqPtr->SetCompletionCallback([&] {
        curIterationNet2++;
        std::cout << "Completed " << curIterationNet2 << " async request execution for network2\n";
        if (curIterationNet2 < static_cast<size_t>(iterationCount)) {
            Blob::Ptr outputBlob;
            std::string output2Name = network2.GetOutputsInfo().begin()->first;
            ASSERT_NO_THROW(outputBlob = network2InferReqPtr->GetBlob(output2Name));
            network2InferReqPtr->StartAsync();
        } else {
            condVar.notify_one();
        }
    });

    std::cout << "Start inference (" << iterationCount << " asynchronous executions) for network1" << std::endl;
    network1InferReqPtr->StartAsync();
    std::cout << "Start inference (" << iterationCount << " asynchronous executions) for network2" << std::endl;
    network2InferReqPtr->StartAsync();

    std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
    condVar.wait(lock, [&] {
        return curIterationNetwork1 == static_cast<size_t>(iterationCount) &&
               curIterationNet2 == static_cast<size_t>(iterationCount);
    });
}

INSTANTIATE_TEST_CASE_P(preprocessing, VpuPreprocessingWithTwoNetworksTests,
    Values(std::make_tuple("/KMB_models/BLOBS/mobilenet-v2/input-1080x1080-nv12.bin", 1080, 1080,
               "/KMB_models/BLOBS/tiny-yolo-v2/input-1920x1080-nv12.bin", 1920, 1080),
        std::make_tuple("/KMB_models/BLOBS/mobilenet-v2/input-1920x1080-nv12.bin", 1920, 1080,
            "/KMB_models/BLOBS/tiny-yolo-v2/input-1920x1080-nv12.bin", 1920, 1080),
        std::make_tuple("/KMB_models/BLOBS/mobilenet-v2/input-228x228-nv12.bin", 228, 228,
            "/KMB_models/BLOBS/tiny-yolo-v2/input-1920x1080-nv12.bin", 1920, 1080)));

TEST_F(vpuLayersTests, allocateNV12WithNative) {
    InferenceEngine::ExecutableNetwork network1;
    std::string network1Path = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";
    ASSERT_NO_THROW(network1 = core->ImportNetwork(network1Path, deviceName, {}));

    ASSERT_EQ(1, network1.GetInputsInfo().size());

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> nativeAllocator = buildAllocator("NATIVE");

    InferenceEngine::InferRequest::Ptr network1InferReqPtr;
    network1InferReqPtr = network1.CreateInferRequestPtr();

    ConstInputsDataMap inputInfo1 = network1.GetInputsInfo();
    std::string input1_name = inputInfo1.begin()->first;
    std::string input1Path = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input-cat-1080x1080-nv12.bin";
    setNV12Preproc(input1_name, input1Path, *network1InferReqPtr, nativeAllocator, 1080, 1080);

    ASSERT_NO_THROW(network1InferReqPtr->Infer());
    ASSERT_EQ(1, network1.GetOutputsInfo().size());

    ConstOutputsDataMap outputInfo;
    ASSERT_NO_THROW(outputInfo = network1.GetOutputsInfo());
    std::string firstOutputName = outputInfo.begin()->first;

    Blob::Ptr outputBlob = toFP32(network1InferReqPtr->GetBlob(firstOutputName));

    TensorDesc outputBlobTensorDesc = outputBlob->getTensorDesc();

    std::string referenceOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output-cat-1080x1080-nv12.bin";
    Blob::Ptr referenceOutputBlob;
    ASSERT_NO_THROW(
        referenceOutputBlob = vpu::KmbPlugin::utils::fromBinaryFile(referenceOutputFilePath, outputBlobTensorDesc));

    const size_t NUMBER_OF_CLASSES = 1;
    ASSERT_NO_THROW(compareTopClasses(outputBlob, referenceOutputBlob, NUMBER_OF_CLASSES));
}

TEST_F(vpuLayersTests, allocateNV12TwoImages) {
    InferenceEngine::ExecutableNetwork network1;
    std::string network1Path = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";
    ASSERT_NO_THROW(network1 = core->ImportNetwork(network1Path, deviceName, {}));

    ASSERT_EQ(1, network1.GetInputsInfo().size());

    ConstInputsDataMap inputInfo1 = network1.GetInputsInfo();

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> nativeAllocator = buildAllocator("NATIVE");

    InferenceEngine::InferRequest::Ptr commonInferReqPtr;
    commonInferReqPtr = network1.CreateInferRequestPtr();

    std::string input1_name = inputInfo1.begin()->first;

    std::string inputCatPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input-cat-1080x1080-nv12.bin";
    setNV12Preproc(input1_name, inputCatPath, *commonInferReqPtr, nativeAllocator, 1080, 1080);

    ASSERT_NO_THROW(commonInferReqPtr->Infer());
    ASSERT_EQ(1, network1.GetOutputsInfo().size());

    ConstOutputsDataMap outputInfo;
    ASSERT_NO_THROW(outputInfo = network1.GetOutputsInfo());
    std::string firstOutputName = outputInfo.begin()->first;

    Blob::Ptr catOutputBlob = toFP32(commonInferReqPtr->GetBlob(firstOutputName));

    TensorDesc outputBlobTensorDesc = catOutputBlob->getTensorDesc();

    std::string catOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output-cat-1080x1080-nv12.bin";
    Blob::Ptr catOutputContentBlob;
    ASSERT_NO_THROW(
        catOutputContentBlob = vpu::KmbPlugin::utils::fromBinaryFile(catOutputFilePath, outputBlobTensorDesc));

    Blob::Ptr catReferenceOutputBlob = toFP32(catOutputContentBlob);

    const size_t NUMBER_OF_CLASSES = 1;
    ASSERT_NO_THROW(compareTopClasses(toFP32(catOutputBlob), catReferenceOutputBlob, NUMBER_OF_CLASSES));

    // set another image to already allocated chunk
    std::string inputDogPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input-dog-1080x1080-nv12.bin";
    uint8_t* inputMemPtr = reinterpret_cast<uint8_t*>(nativeAllocator->getAllocatedChunkByIndex(0));
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::readNV12FileHelper(inputDogPath, (1080 * 1080 * 3) / 2, inputMemPtr, 0));

    ASSERT_NO_THROW(commonInferReqPtr->Infer());

    Blob::Ptr dogOutputBlob = toFP32(commonInferReqPtr->GetBlob(firstOutputName));

    std::string dogOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output-dog-1080x1080-nv12.bin";
    Blob::Ptr dogReferenceOutputBlob;
    ASSERT_NO_THROW(
        dogReferenceOutputBlob = vpu::KmbPlugin::utils::fromBinaryFile(dogOutputFilePath, outputBlobTensorDesc));

    ASSERT_NO_THROW(compareTopClasses(dogOutputBlob, dogReferenceOutputBlob, NUMBER_OF_CLASSES));
}

TEST_F(vpuLayersTests, allocateNV12TwoImagesGetBlob) {
    InferenceEngine::ExecutableNetwork network1;
    std::string network1Path = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";
    ASSERT_NO_THROW(network1 = core->ImportNetwork(network1Path, deviceName, {}));

    ASSERT_EQ(1, network1.GetInputsInfo().size());

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> nativeAllocator = buildAllocator("NATIVE");

    InferenceEngine::InferRequest::Ptr commonInferReqPtr;
    commonInferReqPtr = network1.CreateInferRequestPtr();

    ConstInputsDataMap inputInfo1 = network1.GetInputsInfo();
    std::string input1_name = inputInfo1.begin()->first;

    std::string inputCatPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input-cat-1080x1080-nv12.bin";
    setNV12Preproc(input1_name, inputCatPath, *commonInferReqPtr, nativeAllocator, 1080, 1080);

    ASSERT_NO_THROW(commonInferReqPtr->Infer());
    ASSERT_EQ(1, network1.GetOutputsInfo().size());

    ConstOutputsDataMap outputInfo;
    ASSERT_NO_THROW(outputInfo = network1.GetOutputsInfo());
    std::string firstOutputName = outputInfo.begin()->first;

    Blob::Ptr catOutputBlob = toFP32(commonInferReqPtr->GetBlob(firstOutputName));

    TensorDesc outputBlobTensorDesc = catOutputBlob->getTensorDesc();

    std::string catOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output-cat-1080x1080-nv12.bin";
    Blob::Ptr catOutputContentBlob;
    ASSERT_NO_THROW(
        catOutputContentBlob = vpu::KmbPlugin::utils::fromBinaryFile(catOutputFilePath, outputBlobTensorDesc));

    Blob::Ptr catReferenceOutputBlob = catOutputContentBlob;

    const size_t NUMBER_OF_CLASSES = 1;
    ASSERT_NO_THROW(compareTopClasses(catOutputBlob, catReferenceOutputBlob, NUMBER_OF_CLASSES));

    // set another image via GetBlob
    std::string inputDogPath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/input-dog-1080x1080-nv12.bin";

    IInferRequest::Ptr inferReqInterfacePtr = static_cast<IInferRequest::Ptr&>(*commonInferReqPtr);
    Blob::Ptr dogInputBlobPtr;
    ResponseDesc resp;
    inferReqInterfacePtr->GetBlob(input1_name.c_str(), dogInputBlobPtr, &resp);
    NV12Blob::Ptr dogNV12blobPtr = as<NV12Blob>(dogInputBlobPtr);
    Blob::Ptr& dogYPlane = dogNV12blobPtr->y();
    Blob::Ptr& dogUVPlane = dogNV12blobPtr->uv();
    ASSERT_NO_THROW(
        vpu::KmbPlugin::utils::readNV12FileHelper(inputDogPath, 1080 * 1080, dogYPlane->buffer().as<uint8_t*>(), 0));
    ASSERT_NO_THROW(vpu::KmbPlugin::utils::readNV12FileHelper(
        inputDogPath, 1080 * 1080 / 2, dogUVPlane->buffer().as<uint8_t*>(), 1080 * 1080));

    ASSERT_NO_THROW(commonInferReqPtr->Infer());

    Blob::Ptr dogOutputBlob = toFP32(commonInferReqPtr->GetBlob(firstOutputName));

    std::string dogOutputFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/output-dog-1080x1080-nv12.bin";
    Blob::Ptr dogReferenceOutputBlob;
    ASSERT_NO_THROW(
        dogReferenceOutputBlob = vpu::KmbPlugin::utils::fromBinaryFile(dogOutputFilePath, outputBlobTensorDesc));

    ASSERT_NO_THROW(compareTopClasses(dogOutputBlob, dogReferenceOutputBlob, NUMBER_OF_CLASSES));
}

class VpuPreprocessingConfigAndInferTests :
    public vpuLayersTests,
    public testing::WithParamInterface<std::tuple<const char*, const char*>> {
protected:
    std::map<std::string, std::string> _config;

public:
    void setConfigAndInfer() {
        std::string key, value;
        std::tie(key, value) = GetParam();

        std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/tiny-yolo-v2.blob";

        std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
            buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

        InferenceEngine::ExecutableNetwork importedNetwork;

        _config[key] = value;
        ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName, _config));

        ConstInputsDataMap inputInfo = importedNetwork.GetInputsInfo();

        InferenceEngine::InferRequest inferRequest;
        ASSERT_NO_THROW(inferRequest = importedNetwork.CreateInferRequest());

        inputInfo = importedNetwork.GetInputsInfo();
        std::string input_name = inputInfo.begin()->first;

        setRandomNV12(input_name, inferRequest, kmbAllocator, 1920, 1080);

        ASSERT_NO_THROW(inferRequest.Infer());
    }
};
TEST_P(VpuPreprocessingConfigAndInferTests, setConfigAndInfer) { setConfigAndInfer(); }

class VpuPreprocessingConfigAndInferTestsSipp : public VpuPreprocessingConfigAndInferTests {
public:
    VpuPreprocessingConfigAndInferTestsSipp() {
        // All Sipp-related config options require Sipp to be enabled
        _config["VPU_KMB_USE_SIPP"] = "YES";
    }
};
TEST_P(VpuPreprocessingConfigAndInferTestsSipp, setConfigAndInfer) { setConfigAndInfer(); }

class VpuPreprocessingConfigTests :
    public vpuLayersTests,
    public testing::WithParamInterface<std::tuple<const char*, const char*, bool>> {};
TEST_P(VpuPreprocessingConfigTests, setConfigAndCheck) {
    std::string key, value;
    bool valid;
    std::tie(key, value, valid) = GetParam();
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";

    InferenceEngine::ExecutableNetwork importedNetwork;

    std::map<std::string, std::string> config;
    config[key] = value;

    if (valid) {
        ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName, config));
    } else {
        ASSERT_ANY_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName, config));
    }
}

TEST_F(VpuPreprocessingTests, setConfigForTwoNetworks) {
    InferenceEngine::ExecutableNetwork network1;
    std::string network1Path = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/tiny-yolo-v2.blob";
    std::map<std::string, std::string> config1;
    config1["VPU_KMB_PREPROCESSING_SHAVES"] = "4";
    config1["VPU_KMB_PREPROCESSING_LPI"] = "4";
    ASSERT_NO_THROW(network1 = core->ImportNetwork(network1Path, deviceName, config1));

    InferenceEngine::ExecutableNetwork network2;
    std::string network2Path = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";
    std::map<std::string, std::string> config2;
    config2["VPU_KMB_PREPROCESSING_SHAVES"] = "2";
    config2["VPU_KMB_PREPROCESSING_LPI"] = "8";
    ASSERT_NO_THROW(network2 = core->ImportNetwork(network2Path, deviceName, config2));

    std::cout << "Created networks\n";

    ASSERT_EQ(1, network1.GetInputsInfo().size());
    ASSERT_EQ(1, network2.GetInputsInfo().size());
    std::cout << "Input info is OK\n";

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    InferenceEngine::InferRequest::Ptr network1InferReqPtr;
    network1InferReqPtr = network1.CreateInferRequestPtr();

    ConstInputsDataMap inputInfo1 = network1.GetInputsInfo();
    ConstInputsDataMap inputInfo2 = network2.GetInputsInfo();
    std::string input1_name = inputInfo1.begin()->first;
    std::string input1Path = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/input-228x228-nv12.bin";
    setNV12Preproc(input1_name, input1Path, *network1InferReqPtr, kmbAllocator, 228, 228);

    InferenceEngine::InferRequest::Ptr network2InferReqPtr;
    network2InferReqPtr = network2.CreateInferRequestPtr();

    std::string input2_name = inputInfo2.begin()->first;
    std::string input2Path = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/input-228x228-nv12.bin";
    setNV12Preproc(input2_name, input2Path, *network2InferReqPtr, kmbAllocator, 228, 228);

    std::cout << "Created inference requests\n";

    ASSERT_EQ(1, network1.GetOutputsInfo().size());
    ASSERT_EQ(1, network2.GetOutputsInfo().size());
    std::cout << "Output info is OK\n";

    const auto iterationCount = 5;
    size_t curIterationNetwork1 = 0;
    size_t curIterationNet2 = 0;
    std::condition_variable condVar;

    network1InferReqPtr->SetCompletionCallback([&] {
        curIterationNetwork1++;
        std::cout << "Completed " << curIterationNetwork1 << " async request execution for network1\n";
        if (curIterationNetwork1 < static_cast<size_t>(iterationCount)) {
            Blob::Ptr outputBlob;
            std::string output1Name = network1.GetOutputsInfo().begin()->first;
            ASSERT_NO_THROW(outputBlob = network1InferReqPtr->GetBlob(output1Name));
            network1InferReqPtr->StartAsync();
        } else {
            condVar.notify_one();
        }
    });
    network2InferReqPtr->SetCompletionCallback([&] {
        curIterationNet2++;
        std::cout << "Completed " << curIterationNet2 << " async request execution for network1\n";
        if (curIterationNet2 < static_cast<size_t>(iterationCount)) {
            Blob::Ptr outputBlob;
            std::string output2Name = network2.GetOutputsInfo().begin()->first;
            ASSERT_NO_THROW(outputBlob = network2InferReqPtr->GetBlob(output2Name));
            network2InferReqPtr->StartAsync();
        } else {
            condVar.notify_one();
        }
    });

    std::cout << "Start inference (" << iterationCount << " asynchronous executions) for network1" << std::endl;
    network1InferReqPtr->StartAsync();
    std::cout << "Start inference (" << iterationCount << " asynchronous executions) for network2" << std::endl;
    network2InferReqPtr->StartAsync();

    std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
    condVar.wait(lock, [&] {
        return curIterationNetwork1 == static_cast<size_t>(iterationCount) &&
               curIterationNet2 == static_cast<size_t>(iterationCount);
    });
}

TEST_F(VpuPreprocessingTests, setConfigAndCheckNumShaves) {
    std::string modelFilePath = ModelsPath() + "/KMB_models/BLOBS/resnet-50/resnet-50.blob";

    InferenceEngine::ExecutableNetwork importedNetwork;

    std::map<std::string, std::string> config;
    config["VPU_KMB_PREPROCESSING_SHAVES"] = "8";
    config["VPU_KMB_PREPROCESSING_LPI"] = "4";

    ASSERT_NO_THROW(importedNetwork = core->ImportNetwork(modelFilePath, deviceName, config));
    importedNetwork.SetConfig({{"VPU_KMB_PREPROCESSING_SHAVES", "6"}, {"VPU_KMB_PREPROCESSING_LPI", "2"}});
    InferenceEngine::Parameter param1 = importedNetwork.GetConfig("VPU_KMB_PREPROCESSING_SHAVES");
    InferenceEngine::Parameter param2 = importedNetwork.GetConfig("VPU_KMB_PREPROCESSING_LPI");
    std::cout << "Config key: VPU_KMB_PREPROCESSING_SHAVES; value: " << param1.as<std::string>() << std::endl;
    std::cout << "Config key: VPU_KMB_PREPROCESSING_LPI; value: " << param2.as<std::string>() << std::endl;
}

// TODO consider re-using twoNetworksWithPreprocessing instead of duplicating it
using VpuPreprocessingStressTests = KmbYoloV2NetworkTest;

// [Track number: S#35173, S#35231]
TEST_F(VpuPreprocessingStressTests, DISABLED_twoNetworksHDImage1000Iterations) {
    if (!KmbTestBase::RUN_INFER) {
        SKIP();
    }
    Core ie;
    InferenceEngine::ExecutableNetwork network1;
    std::string network1Path = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";
    ASSERT_NO_THROW(network1 = ie.ImportNetwork(network1Path, "KMB", {}));

    std::string network2Path = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/tiny-yolo-v2.blob";
    InferenceEngine::ExecutableNetwork network2;
    ASSERT_NO_THROW(network2 = ie.ImportNetwork(network2Path, "KMB", {}));

    std::cout << "Created networks\n";

    ASSERT_EQ(1, network1.GetInputsInfo().size());
    ASSERT_EQ(1, network2.GetInputsInfo().size());
    std::cout << "Input info is OK\n";

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    InferenceEngine::InferRequest::Ptr network1InferReqPtr;
    network1InferReqPtr = network1.CreateInferRequestPtr();

    ConstInputsDataMap inputInfo1 = network1.GetInputsInfo();
    ConstInputsDataMap inputInfo2 = network2.GetInputsInfo();

    const size_t imgWidth = 1920;
    const size_t imgHeight = 1080;

    std::string input1_name = inputInfo1.begin()->first;
    std::string input1Path = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-1-1920x1080-nv12.bin";
    setNV12Preproc(input1_name, input1Path, *network1InferReqPtr, kmbAllocator, imgWidth, imgHeight);

    InferenceEngine::InferRequest::Ptr network2InferReqPtr;
    network2InferReqPtr = network2.CreateInferRequestPtr();

    std::string input2_name = inputInfo2.begin()->first;
    std::string input2Path = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-2-1920x1080-nv12.bin";
    setNV12Preproc(input2_name, input2Path, *network2InferReqPtr, kmbAllocator, imgWidth, imgHeight);

    std::cout << "Created inference requests\n";

    ASSERT_EQ(1, network1.GetOutputsInfo().size());
    ASSERT_EQ(1, network2.GetOutputsInfo().size());
    std::cout << "Output info is OK\n";

    const auto iterationCount = 1000;
    size_t curIterationNetwork1 = 0;
    size_t curIterationNetwork2 = 0;
    std::condition_variable condVar;

    network1InferReqPtr->SetCompletionCallback([&] {
        curIterationNetwork1++;
        std::cout << "Completed " << curIterationNetwork1 << " async request execution for network 1" << std::endl;
        if (curIterationNetwork1 < static_cast<size_t>(iterationCount)) {
            Blob::Ptr outputBlob;
            std::string output1Name = network1.GetOutputsInfo().begin()->first;
            ASSERT_NO_THROW(outputBlob = network1InferReqPtr->GetBlob(output1Name));
            network1InferReqPtr->StartAsync();
        } else {
            condVar.notify_one();
        }
    });
    const size_t BBOX_PRINT_INTERVAL = 100;
    network2InferReqPtr->SetCompletionCallback([&] {
        curIterationNetwork2++;
        std::cout << "Completed " << curIterationNetwork2 << " async request execution for network 2" << std::endl;
        if (curIterationNetwork2 < static_cast<size_t>(iterationCount)) {
            Blob::Ptr outputBlob;
            std::string output2Name = network2.GetOutputsInfo().begin()->first;
            ASSERT_NO_THROW(outputBlob = network2InferReqPtr->GetBlob(output2Name));

            if (curIterationNetwork2 % BBOX_PRINT_INTERVAL == 0) {
                float confThresh = 0.4f;
                bool isTiny = true;
                auto actualOutput = utils::parseYoloOutput(outputBlob, imgWidth, imgHeight, confThresh, isTiny);
                std::cout << "BBox Top:" << std::endl;
                for (size_t i = 0; i < actualOutput.size(); ++i) {
                    const auto& bb = actualOutput[i];
                    std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right
                              << " " << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
                }
            }
            network2InferReqPtr->StartAsync();
        } else {
            condVar.notify_one();
        }
    });

    std::cout << "Start inference (" << iterationCount << " asynchronous executions) for network1" << std::endl;
    network1InferReqPtr->StartAsync();
    std::cout << "Start inference (" << iterationCount << " asynchronous executions) for network2" << std::endl;
    network2InferReqPtr->StartAsync();

    std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
    condVar.wait(lock, [&] {
        return curIterationNetwork1 == static_cast<size_t>(iterationCount) &&
               curIterationNetwork2 == static_cast<size_t>(iterationCount);
    });
}

// [Track number: S#35173, S#35231]
TEST_F(VpuPreprocessingStressTests, DISABLED_twoNetworksStressTest) {
    if (!KmbTestBase::RUN_INFER) {
        SKIP();
    }
    Core ie;
    InferenceEngine::ExecutableNetwork network1;
    std::string network1Path = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";
    ASSERT_NO_THROW(network1 = ie.ImportNetwork(network1Path, "KMB", {}));

    std::string network2Path = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/tiny-yolo-v2.blob";
    InferenceEngine::ExecutableNetwork network2;
    ASSERT_NO_THROW(network2 = ie.ImportNetwork(network2Path, "KMB", {}));

    std::cout << "Created networks\n";

    ASSERT_EQ(1, network1.GetInputsInfo().size());
    ASSERT_EQ(1, network2.GetInputsInfo().size());
    std::cout << "Input info is OK\n";

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    InferenceEngine::InferRequest::Ptr network1InferReqPtr;
    network1InferReqPtr = network1.CreateInferRequestPtr();

    ConstInputsDataMap inputInfo1 = network1.GetInputsInfo();
    ConstInputsDataMap inputInfo2 = network2.GetInputsInfo();

    const size_t imgWidth = 1920;
    const size_t imgHeight = 1080;

    std::string input1_name = inputInfo1.begin()->first;
    std::string input1Path = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-1-1920x1080-nv12.bin";
    setNV12Preproc(input1_name, input1Path, *network1InferReqPtr, kmbAllocator, imgWidth, imgHeight);

    InferenceEngine::InferRequest::Ptr network2InferReqPtr;
    network2InferReqPtr = network2.CreateInferRequestPtr();

    std::string input2_name = inputInfo2.begin()->first;
    std::string input2Path = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-2-1920x1080-nv12.bin";
    setNV12Preproc(input2_name, input2Path, *network2InferReqPtr, kmbAllocator, imgWidth, imgHeight);

    std::cout << "Created inference requests\n";

    ASSERT_EQ(1, network1.GetOutputsInfo().size());
    ASSERT_EQ(1, network2.GetOutputsInfo().size());
    std::cout << "Output info is OK\n";

    const std::chrono::system_clock::time_point timeLimit =
        std::chrono::system_clock::now() + std::chrono::seconds(10 * 60);  // 10 minutes
    size_t curIterationNetwork1 = 0;
    size_t curIterationNetwork2 = 0;
    std::condition_variable condVar;

    network1InferReqPtr->SetCompletionCallback([&] {
        curIterationNetwork1++;
        std::cout << "Completed " << curIterationNetwork1 << " async request execution for network 1" << std::endl;
        if (std::chrono::system_clock::now() < timeLimit) {
            Blob::Ptr outputBlob;
            std::string output1Name = network1.GetOutputsInfo().begin()->first;
            ASSERT_NO_THROW(outputBlob = network1InferReqPtr->GetBlob(output1Name));
            network1InferReqPtr->StartAsync();
        } else {
            condVar.notify_one();
        }
    });

    const size_t BBOX_PRINT_INTERVAL = 100;
    network2InferReqPtr->SetCompletionCallback([&] {
        curIterationNetwork2++;
        std::cout << "Completed " << curIterationNetwork2 << " async request execution for network 2" << std::endl;
        if (std::chrono::system_clock::now() < timeLimit) {
            Blob::Ptr outputBlob;
            std::string output2Name = network2.GetOutputsInfo().begin()->first;
            ASSERT_NO_THROW(outputBlob = network2InferReqPtr->GetBlob(output2Name));

            if (curIterationNetwork2 % BBOX_PRINT_INTERVAL == 0) {
                float confThresh = 0.4f;
                bool isTiny = true;
                auto actualOutput = utils::parseYoloOutput(outputBlob, imgWidth, imgHeight, confThresh, isTiny);
                std::cout << "BBox Top:" << std::endl;
                for (size_t i = 0; i < actualOutput.size(); ++i) {
                    const auto& bb = actualOutput[i];
                    std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right
                              << " " << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
                }
            }
            network2InferReqPtr->StartAsync();
        } else {
            condVar.notify_one();
        }
    });

    std::cout << "Start inference for network1" << std::endl;
    network1InferReqPtr->StartAsync();
    std::cout << "Start inference for network2" << std::endl;
    network2InferReqPtr->StartAsync();

    std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
    condVar.wait(lock, [&] {
        return std::chrono::system_clock::now() >= timeLimit;
    });
}

// [Track number: S#35173, S#35231]
TEST_F(VpuPreprocessingStressTests, DISABLED_detectClassify4Threads) {
    if (!KmbTestBase::RUN_INFER) {
        SKIP();
    }
    Core ie;

    std::string detectNetworkPath = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/tiny-yolo-v2.blob";
    InferenceEngine::ExecutableNetwork detectionNetwork = ie.ImportNetwork(detectNetworkPath, "KMB", {});

    std::string classifyNetworkPath = ModelsPath() + "/KMB_models/BLOBS/mobilenet-v2/mobilenet-v2.blob";
    InferenceEngine::ExecutableNetwork classificationNetwork = ie.ImportNetwork(classifyNetworkPath, "KMB", {});

    std::shared_ptr<vpu::KmbPlugin::utils::VPUAllocator> kmbAllocator =
        buildAllocator(std::getenv("IE_VPU_KMB_MEMORY_ALLOCATOR_TYPE"));

    std::vector<InferenceEngine::InferRequest::Ptr> detectionRequests;
    std::vector<InferenceEngine::InferRequest::Ptr> classificationRequests;
    ConstInputsDataMap detectInputInfo = detectionNetwork.GetInputsInfo();
    ConstInputsDataMap classifyInputInfo = classificationNetwork.GetInputsInfo();

    const size_t imgWidth = 1920;
    const size_t imgHeight = 1080;
    const size_t maxParallelRequests = 4;

    std::condition_variable condVar;
    const char* iterCountStr = std::getenv("IE_VPU_KMB_STRESS_TEST_ITER_COUNT");
    const size_t iterationCount = (iterCountStr != nullptr) ? std::atoi(iterCountStr) : 10;
    std::vector<size_t> iterationVec(maxParallelRequests, 0);

    std::string detectInputName = detectInputInfo.begin()->first;
    std::string detectInputPath = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-1-1920x1080-nv12.bin";
    std::string classifyInputName = classifyInputInfo.begin()->first;
    std::string classifyInputPath = ModelsPath() + "/KMB_models/BLOBS/tiny-yolo-v2/cars-frame-2-1920x1080-nv12.bin";

    std::string detectOutputName = detectionNetwork.GetOutputsInfo().begin()->first;
    std::string classifyOutputName = classificationNetwork.GetOutputsInfo().begin()->first;

    for (size_t requestNum = 0; requestNum < maxParallelRequests; requestNum++) {
        InferenceEngine::InferRequest::Ptr detectInferReq = detectionNetwork.CreateInferRequestPtr();
        setNV12Preproc(detectInputName, detectInputPath, *detectInferReq, kmbAllocator, imgWidth, imgHeight);

        InferenceEngine::InferRequest::Ptr classifyInferReq = classificationNetwork.CreateInferRequestPtr();
        setNV12Preproc(classifyInputName, classifyInputPath, *classifyInferReq, kmbAllocator, imgWidth, imgHeight);

        auto detectCallback = [requestNum, &iterationVec, iterationCount, &detectionRequests, &classificationRequests,
                                  detectOutputName, &condVar, maxParallelRequests](void) -> void {
            size_t reqId = requestNum;
            iterationVec[reqId]++;
            std::cout << "Completed " << iterationVec[reqId] << " async request ID " << reqId << std::endl;
            if (iterationVec[reqId] < iterationCount) {
                Blob::Ptr detectOutputBlob = detectionRequests.at(reqId)->GetBlob(detectOutputName);

                float confThresh = 0.4f;
                bool isTiny = true;
                auto actualOutput = utils::parseYoloOutput(detectOutputBlob, imgWidth, imgHeight, confThresh, isTiny);
                std::cout << "BBox Top:" << std::endl;
                for (size_t i = 0; i < actualOutput.size(); ++i) {
                    const auto& bb = actualOutput[i];
                    std::cout << i << " : " << bb.idx << " : [(" << bb.left << " " << bb.top << "), (" << bb.right
                              << " " << bb.bottom << ")] : " << bb.prob * 100 << "%" << std::endl;
                }
                for (size_t classReqIdx = 0; classReqIdx < maxParallelRequests; classReqIdx++) {
                    classificationRequests.at(classReqIdx)->StartAsync();
                }
                detectionRequests.at(reqId)->StartAsync();
            } else {
                condVar.notify_one();
            }
        };

        auto classifyCallback = [requestNum, &classificationRequests, classifyOutputName](void) -> void {
            size_t reqId = requestNum;
            Blob::Ptr classifyOutputBlob = classificationRequests.at(reqId)->GetBlob(classifyOutputName);
            const float* bufferRawPtr = classifyOutputBlob->cbuffer().as<const float*>();
            std::vector<float> outputData(classifyOutputBlob->size());
            std::memcpy(outputData.data(), bufferRawPtr, outputData.size());
            std::vector<float>::iterator maxElt = std::max_element(outputData.begin(), outputData.end());
            std::cout << "Top class: " << std::distance(outputData.begin(), maxElt) << std::endl;
        };
        detectInferReq->SetCompletionCallback(detectCallback);
        detectionRequests.push_back(detectInferReq);
        classifyInferReq->SetCompletionCallback(classifyCallback);
        classificationRequests.push_back(classifyInferReq);
    }

    for (const auto& detectReq : detectionRequests) {
        detectReq->StartAsync();
    }

    std::mutex mutex;
    std::unique_lock<std::mutex> lock(mutex);
    condVar.wait(lock, [&] {
        bool allRequestsFinished = true;
        for (size_t iterNum : iterationVec) {
            if (iterNum < iterationCount) {
                allRequestsFinished = false;
            }
        }
        return allRequestsFinished;
    });
}

const static std::vector<preprocessingType> preprocTypes = {PT_RESIZE, PT_NV12};

INSTANTIATE_TEST_CASE_P(preprocessing, VpuPreprocessingTestsWithParam, ::testing::ValuesIn(preprocTypes));

using namespace testing;
INSTANTIATE_TEST_CASE_P(preprocessingShaves, VpuPreprocessingConfigAndInferTestsSipp,
    Combine(Values("VPU_KMB_PREPROCESSING_SHAVES"), Values("4", "6")));

INSTANTIATE_TEST_CASE_P(preprocessingLpi, VpuPreprocessingConfigAndInferTestsSipp,
    Combine(Values("VPU_KMB_PREPROCESSING_LPI"), Values("4", "8")));

INSTANTIATE_TEST_CASE_P(DISABLED_preprocessingM2I, VpuPreprocessingConfigAndInferTests,
    Combine(Values("VPU_KMB_USE_M2I"), Values("YES", "NO")));

INSTANTIATE_TEST_CASE_P(preprocessing, VpuPreprocessingConfigTests,
    Values(std::make_tuple("VPU_KMB_PREPROCESSING_SHAVES", "8", true),
        std::make_tuple("VPU_KMB_PREPROCESSING_SHAVES", "seventy one", false),
        std::make_tuple("VPU_KMB_PREPROCESSING_LPI", "16", true),
        std::make_tuple("VPU_KMB_PREPROCESSING_LPI", "3", false),
        std::make_tuple("VPU_KMB_PREPROCESSING_LPI", "seventeen", false),
        std::make_tuple("VPU_KMB_USE_M2I", "YES", true), std::make_tuple("VPU_KMB_USE_M2I", "USE", false)));

#endif
