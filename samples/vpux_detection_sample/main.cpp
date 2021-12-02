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

#include <algorithm>
#include <fstream>
#include <limits>
#include <list>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <inference_engine.hpp>

#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "detection_sample_yolov2tiny.h"
#include "region_yolov2tiny.h"

#include <format_reader_ptr.h>
#include <ie_compound_blob.h>

#include <vpumgr.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <yolo_helpers.hpp>

#include "vpux/utils/IE/blob.hpp"
#include "openvino/openvino.hpp"

using namespace ov;

enum preprocessingType { PT_RESIZE, PT_NV12 };

class VPUAllocator {
public:
    VPUAllocator() {
    }
    virtual ~VPUAllocator();
    void* allocate(size_t requestedSize);

private:
    std::list<std::tuple<int, void*, size_t>> _memChunks;
    static uint32_t _pageSize;
};

uint32_t VPUAllocator::_pageSize = getpagesize();

static uint32_t calculateRequiredSize(uint32_t blobSize, uint32_t pageSize) {
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

    void* virtAddr = mmap(0, requiredBlobSize, PROT_READ | PROT_WRITE, MAP_SHARED, fileDesc, 0);
    if (virtAddr == MAP_FAILED) {
        throw std::runtime_error("VPUAllocator::allocate: mmap failed");
    }
    std::tuple<int, void*, size_t> memChunk(fileDesc, virtAddr, requiredBlobSize);
    _memChunks.push_back(memChunk);

    return virtAddr;
}

VPUAllocator::~VPUAllocator() {
    for (const std::tuple<int, void*, size_t>& chunk : _memChunks) {
        int fileDesc = std::get<0>(chunk);
        void* virtAddr = std::get<1>(chunk);
        size_t allocatedSize = std::get<2>(chunk);
        vpusmm_unimport_dmabuf(fileDesc);
        munmap(virtAddr, allocatedSize);
        close(fileDesc);
    }
}

static std::pair<ov::runtime::Tensor, ov::runtime::Tensor>
fromNV12File(const std::string& filePath,
             size_t imageWidth,
             size_t imageHeight,
             VPUAllocator& allocator) {
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

    uint8_t* imageData = reinterpret_cast<uint8_t*>(allocator.allocate(fileSize));
    if (!imageData) {
        throw std::runtime_error("fromNV12File: failed to allocate memory");
    }

    fileReader.read(reinterpret_cast<char*>(imageData), fileSize);
    fileReader.close();

    ov::runtime::Tensor tensorY = ov::runtime::Tensor(ov::element::u8,
                                                      {1, 1, imageHeight, imageWidth},
                                                      imageData);

    const size_t offset = imageHeight * imageWidth;
    ov::runtime::Tensor tensorUV = ov::runtime::Tensor(ov::element::u8,
                                                      {1, 2, imageHeight / 2, imageWidth / 2},
                                                      imageData + offset);
    return {tensorY, tensorUV};
}

// static void setPreprocForInputBlob(const std::string& inputName, const std::string& inputFilePath,
//                                    InferenceEngine::InferRequest& inferRequest, VPUAllocator& allocator) {
//     Blob::Ptr inputBlobNV12;
//     const size_t expectedWidth = FLAGS_iw;
//     const size_t expectedHeight = FLAGS_ih;
//     inputBlobNV12 = fromNV12File(inputFilePath, expectedWidth, expectedHeight, allocator);
//     PreProcessInfo preprocInfo = inferRequest.GetPreProcess(inputName);
//     preprocInfo.setResizeAlgorithm(RESIZE_BILINEAR);
//     preprocInfo.setColorFormat(ColorFormat::NV12);
//     inferRequest.SetBlob(inputName, inputBlobNV12, preprocInfo);
// }

ov::runtime::Tensor deQuantize(const ov::runtime::Tensor &quantTensor,
                               float scale,
                               uint8_t zeroPoint) {
    auto outputTensor = ov::runtime::Tensor(ov::element::f32, quantTensor.get_shape());
    const auto* quantRaw = quantTensor.data<const uint8_t>();
    float* outRaw = outputTensor.data<float>();

    for (size_t pos = 0; pos < quantTensor.get_size(); pos++) {
        outRaw[pos] = (quantRaw[pos] - zeroPoint) * scale;
    }

    return outputTensor;
}

ov::runtime::Tensor convertF16ToF32(const ov::runtime::Tensor& inputTensor) {
    auto outputTensor = ov::runtime::Tensor(ov::element::f32, inputTensor.get_shape());
    const auto* inRaw = inputTensor.data<float16>();
    float* outRaw = outputTensor.data<float>();

    for (size_t pos = 0; pos < outputTensor.get_size(); pos++) {
        outRaw[pos] = inRaw[pos];
    }

    return outputTensor;
}

//===========================================================================

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    // -------- Parsing and validation of input arguments --------
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

    if (fileExt(FLAGS_i) == "yuv") {
        if (FLAGS_iw == 0) {
            throw std::logic_error("Parameter -iw is not set");
        }

        if (FLAGS_ih == 0) {
            throw std::logic_error("Parameter -ih is not set");
        }
    }

    return true;
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
int main(int argc, char* argv[]) {
    try {
        VPUAllocator kmbAllocator;

        // -------- Get OpenVINO runtime version --------
        slog::info << ov::get_openvino_version() << slog::endl;

        // -------- Parsing and validation of input arguments --------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** path to the processed image **/
        std::string imagePath = FLAGS_i;

        /** path to pre-compiled blob **/
        std::string blobPath = FLAGS_m;

        // -------- 1. Initialize OpenVINO Runtime Core --------
        slog::info << "Creating Inference Engine" << slog::endl;
        ov::runtime::Core core;

        // -------- 2. Read pre-compiled blob --------
        ov::runtime::ExecutableNetwork executable_network = [&]() {
            slog::info << "Loading blob:\t" << blobPath << slog::endl;
            std::ifstream blob_stream(blobPath, std::ios::binary);
            return core.import_model(blob_stream, "VPUX", {});
        }();

        OPENVINO_ASSERT(executable_network.outputs().size() == 1,
                        "Sample supports models with 1 output only");

        // -------- 3. Set up input --------
        slog::info << "Processing input tensor" << slog::endl;

        const auto inputs = executable_network.inputs();
        OPENVINO_ASSERT(!inputs.empty(), "inputs are empty");

        std::size_t originalImageWidth = 0;
        std::size_t originalImageHeight = 0;
        std::shared_ptr<uint8_t> imageData(nullptr);

        ov::runtime::Tensor input_tensor;

        bool isYUVInput = fileExt(FLAGS_i) == "yuv";
        if (isYUVInput) {
            // setPreprocForInputBlob(firstInputName, imageFileName, inferRequest, kmbAllocator);
            // originalImageWidth = FLAGS_iw;
            // originalImageHeight = FLAGS_ih;
        } else {
            // Read input image to a tensor and set it to an infer request
            // without resize and layout conversions
            FormatReader::ReaderPtr reader(imagePath.c_str());
            if (reader.get() == nullptr) {
                throw std::logic_error("Image " + imagePath + " cannot be read!");
            }

            originalImageWidth = reader->width();
            originalImageHeight = reader->height();

            /** Image reader is expected to return interlaced (NHWC) BGR image **/
            auto input = *executable_network.inputs().begin();
            ov::element::Type input_type = input.get_tensor().get_element_type();
            ov::Shape input_shape = input.get_shape();

            size_t image_width = input_shape.at(3);
            size_t image_height = input_shape.at(2);
            std::shared_ptr<unsigned char> input_data = reader->getData(image_width, image_height);

            // just wrap image data by ov::runtime::Tensor without allocating of new memory
            input_tensor = ov::runtime::Tensor(input_type, input_shape, input_data.get());
        }

        // -------- 4. Create an infer request --------
        ov::runtime::InferRequest infer_request = executable_network.create_infer_request();
        slog::info << "CreateInferRequest completed successfully" << slog::endl;

        // -------- 5. Prepare input --------
        infer_request.set_input_tensor(input_tensor);

        // -------- 6. Do inference synchronously --------
        infer_request.infer();
        slog::info << "inferRequest completed successfully" << slog::endl;

        // -------- 7. Process output --------
        slog::info << "Processing output tensor" << slog::endl;

        ov::runtime::Tensor output_tensor = infer_request.get_output_tensor();
        if (!output_tensor) {
            throw std::logic_error("Cannot get output tensor from infer_request!");
        }

        // de-Quantization
        int zeroPoint = FLAGS_z;
        if (zeroPoint < std::numeric_limits<uint8_t>::min() || zeroPoint > std::numeric_limits<uint8_t>::max()) {
            slog::warn << "zeroPoint value " << zeroPoint << " overflows byte. Setting default." << slog::endl;
            zeroPoint = DEFAULT_ZERO_POINT;
        }
        float scale = static_cast<float>(FLAGS_s);
        slog::info << "zeroPoint: " << zeroPoint << slog::endl;
        slog::info << "scale: " << scale << slog::endl;

        // Real data layer
        ov::runtime::Tensor regionYoloOutput;
        slog::info << "De-quantize if necessary" << slog::endl;
        if (regionYoloOutput.get_element_type() == ov::element::u8) {
            regionYoloOutput = deQuantize(regionYoloOutput, scale, zeroPoint);
        } else {
            regionYoloOutput = convertF16ToF32(regionYoloOutput);
        }
        slog::info << "De-quantization done" << slog::endl;

        const auto imgWidth = originalImageWidth;
        const auto imgHeight = originalImageHeight;

        bool isTiny = true;
        float confThresh = 0.4;
        auto detectionResult = utils::parseYoloOutput(regionYoloOutput, imgWidth, imgHeight, confThresh, isTiny);

        // Print result.
        std::ostringstream resultString;
        utils::printDetectionBBoxOutputs(detectionResult, resultString, postprocess::YOLOV2_TINY_LABELS);
        slog::info << resultString.str() << slog::endl;

        const auto outFilePath = "./output.dat";
        std::ofstream outFile(outFilePath, std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<const char*>(regionYoloOutput.data()),
                          regionYoloOutput.get_byte_size());
        } else {
            slog::warn << "Failed to open '" << outFilePath << "'" << slog::endl;
        }
        outFile.close();
    } catch (const std::exception& error) {
        slog::err << "" << error.what() << slog::endl;
        return 1;
    } catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
