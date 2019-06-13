// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstring>
#include <thread>
#include <chrono>
#include <inference_engine/blob_factory.hpp>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include "format_reader_ptr.h"

#if defined(__arm__) || defined(_M_ARM) || defined(__aarch64__) || defined(_M_ARM64)
#   define DISABLE_ON_ARM      SKIP() << "Disabled on ARM" << std::endl;
#else
#   define DISABLE_ON_ARM
#endif

using namespace InferenceEngine;

PRETTY_PARAM(kernel, param_size)
PRETTY_PARAM(stride, param_size)
PRETTY_PARAM(pad, param_size)
PRETTY_PARAM(out_channels, int)
PRETTY_PARAM(group, int)

class myriadXHWLayersTests_nightly : public myriadLayersTests_nightly {
public:
    void CheckHWRun() {
        StatusCode st;

        std::map<std::string, InferenceEngineProfileInfo> perfMap;
        ASSERT_NO_THROW(st = _inferRequest->GetPerformanceCounts(perfMap, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        std::vector<std::pair<std::string, InferenceEngineProfileInfo>> perfVec(perfMap.begin(), perfMap.end());
        std::sort(perfVec.begin(), perfVec.end(),
            [=](const std::pair<std::string, InferenceEngineProfileInfo> &pair1,
                const std::pair<std::string, InferenceEngineProfileInfo> &pair2) {
                return pair1.second.execution_index < pair2.second.execution_index;
            });

        size_t maxLayerName = 0u, maxExecType = 0u;
        for (auto it = perfVec.begin(); it != perfVec.end(); ++it) {
            maxLayerName = std::max(maxLayerName, it->first.length());
            maxExecType = std::max(maxExecType, std::strlen(it->second.exec_type));
        }

        size_t indexWidth = 7, nameWidth = maxLayerName + 5, typeWidth = maxExecType + 5, timeWidth = 10;
        size_t totalWidth = indexWidth + nameWidth + typeWidth + timeWidth;

        std::cout << std::endl << "Detailed Per Stage Profile" << std::endl;
        for (size_t i = 0; i < totalWidth; i++)
            std::cout << "=";
        std::cout << std::endl;
        std::cout << std::setw(indexWidth) << std::left << "Index"
                  << std::setw(nameWidth) << std::left << "Name"
                  << std::setw(typeWidth) << std::left << "Type"
                  << std::setw(timeWidth) << std::right << "Time (ms)"
                  << std::endl;
        for (size_t i = 0; i < totalWidth; i++)
            std::cout << "-";
        std::cout << std::endl;

        bool hasHWStage = false;
        long long totalTime = 0;
        for (const auto& p : perfVec) {
            const auto& stageName = p.first;
            const auto& info = p.second;
            if (info.status == InferenceEngineProfileInfo::EXECUTED) {
                std::string stageType(info.exec_type);
                if (stageType.find("MyriadXHw") != std::string::npos) {
                    hasHWStage = true;
                }

                std::cout << std::setw(indexWidth) << std::left << info.execution_index
                          << std::setw(nameWidth) << std::left << stageName
                          << std::setw(typeWidth) << std::left << info.exec_type
                          << std::setw(timeWidth) << std::right << info.realTime_uSec / 1000.0
                          << std::endl;

                totalTime += info.realTime_uSec;
            }
        }

        for (int i = 0; i < totalWidth; i++)
            std::cout << "-";
        std::cout << std::endl;
        std::cout << std::setw(totalWidth / 2) << std::right << "Total inference time:"
                  << std::setw(totalWidth / 2 + 1) << std::right << totalTime / 1000.0
                  << std::endl;
        for (int i = 0; i < totalWidth; i++)
            std::cout << "-";
        std::cout << std::endl;

        EXPECT_TRUE(hasHWStage);
    }

    void RunNetwork(const CNNNetwork& network,
                    bool hwEnabled,
                    const Blob::Ptr& input,
                    Blob::Ptr& output,
                    const char* inputName,
                    const char* outputName,
                    float inputNorm = 1.0f,
                    const std::string& networkConfig = "",
                    const std::string& hwBlackList = "") {
        _inferRequest.reset();
        _exeNetwork.reset();

        StatusCode st;

        ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network,
                                                          {
                                                              {
                                                                  VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION),
                                                                  hwEnabled ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO)
                                                              },
                                                              {
                                                                  VPU_CONFIG_KEY(INPUT_NORM),
                                                                  std::to_string(inputNorm)
                                                              },
                                                              {
                                                                  VPU_CONFIG_KEY(NETWORK_CONFIG),
                                                                  hwEnabled ? networkConfig : ""
                                                              },
                                                              {
                                                                  VPU_CONFIG_KEY(HW_BLACK_LIST),
                                                                  hwBlackList
                                                              },
                                                              {
                                                                  CONFIG_KEY(PERF_COUNT),
                                                                  CONFIG_VALUE(YES)
                                                              }
                                                          },
                                                          &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = _inferRequest->SetBlob(inputName, input, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = _inferRequest->GetBlob(outputName, output, &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    }

    void RunBothCases(Blob::Ptr& swOutput, Blob::Ptr& hwOutput) {
        {
            SCOPED_TRACE("SW");

            SetSeed(DEFAULT_SEED_VALUE);

            _net_reader = CNNNetReader();
            _exeNetwork.reset();
            _inferRequest.reset();
            ASSERT_TRUE(GenerateNetAndInfer(false, false));

            auto outBlob = _outputMap.begin()->second;
            swOutput = make_shared_blob<ie_fp16, const SizeVector>(outBlob->precision(), outBlob->layout(), outBlob->dims());
            swOutput->allocate();
            std::copy_n(outBlob->cbuffer().as<const uint8_t*>(), outBlob->byteSize(), swOutput->buffer().as<uint8_t*>());
        }

        {
            SCOPED_TRACE("HW");

            SetSeed(DEFAULT_SEED_VALUE);

            _net_reader = CNNNetReader();
            _exeNetwork.reset();
            _inferRequest.reset();
            ASSERT_TRUE(GenerateNetAndInfer(true, false));

            auto outBlob = _outputMap.begin()->second;
            hwOutput = make_shared_blob<ie_fp16, const SizeVector>(outBlob->precision(), outBlob->layout(), outBlob->dims());
            hwOutput->allocate();
            std::copy_n(outBlob->cbuffer().as<const uint8_t*>(), outBlob->byteSize(), hwOutput->buffer().as<uint8_t*>());

            CheckHWRun();
        }
    }
};

//
// Networks
//

using HwNetworkParams = std::tuple<Precision, Precision>;

class myriadXHWNetworks_nightly :
        public myriadXHWLayersTests_nightly,
        public testing::WithParamInterface<HwNetworkParams> {
public:
    Precision inputPrecision;
    Precision outputPrecision;

    InputInfo::Ptr _inputInfo;
    DataPtr _outputInfo;
    Blob::Ptr _input;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(myriadXHWLayersTests_nightly::SetUp());

        inputPrecision = std::get<0>(GetParam());
        outputPrecision = std::get<1>(GetParam());
    }

    void LoadNetwork(const char* model, const char* inputBlobName, const char* outputBlobName) {
        _net_reader = CNNNetReader();

        std::ostringstream modelFile;
        modelFile << "/" << model << ".xml";

        std::ostringstream weightsFile;
        weightsFile << "/" << model << ".bin";

        std::string modelFilePath = ModelsPath() + modelFile.str();
        std::string weightsFilePath = ModelsPath() + weightsFile.str();

        ASSERT_NO_THROW(_net_reader.ReadNetwork(modelFilePath));
        ASSERT_TRUE(_net_reader.isParseSuccess());
        ASSERT_NO_THROW(_net_reader.ReadWeights(weightsFilePath));

        _inputsInfo = _net_reader.getNetwork().getInputsInfo();
        _inputInfo = _inputsInfo[inputBlobName];
        ASSERT_NE(_inputInfo, nullptr);

        _outputsInfo = _net_reader.getNetwork().getOutputsInfo();
        _outputInfo = _outputsInfo[outputBlobName];
        ASSERT_NE(_outputInfo, nullptr);
    }

    void LoadInput(const char* inputFile, bool swapChannels = false) {
        switch (inputPrecision) {
        case Precision::U8:
            _input = make_shared_blob<uint8_t, const SizeVector>(Precision::U8, Layout::NCHW, _inputInfo->getDims());
            break;
        case Precision::FP16:
            _input = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, _inputInfo->getDims());
            break;
        case Precision::FP32:
            _input = make_shared_blob<float, const SizeVector>(Precision::FP32, Layout::NCHW, _inputInfo->getDims());
            break;
        default:
            FAIL() << "Unsupported precision " << inputPrecision;
        }

        _input->allocate();

        std::ostringstream inputFilePath;
        inputFilePath << get_data_path() << "/" << inputFile;

        FormatReader::ReaderPtr reader(inputFilePath.str().c_str());
        ASSERT_NE(reader.get(), nullptr);

        auto data = reader->getData();

        const auto& dims = _input->getTensorDesc().getDims();
        auto C = dims[1];
        auto H = dims[2];
        auto W = dims[3];

        if (inputPrecision == Precision::U8) {
            auto inputPtr = _input->buffer().as<uint8_t *>();
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int dataC = swapChannels ? C - 1 - c : c;
                        inputPtr[w + h * W + c * W * H] = data.get()[dataC + w * C + h * C * W];
                    }
                }
            }
        } else if (inputPrecision == Precision::FP16) {
            auto inputPtr = _input->buffer().as<ie_fp16 *>();
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int dataC = swapChannels ? C - 1 - c : c;
                        inputPtr[w + h * W + c * W * H] = PrecisionUtils::f32tof16(data.get()[dataC + w * C + h * C * W]);
                    }
                }
            }
        } else if (inputPrecision == Precision::FP32) {
            auto inputPtr = _input->buffer().as<float *>();
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int dataC = swapChannels ? C - 1 - c : c;
                        inputPtr[w + h * W + c * W * H] = data.get()[dataC + w * C + h * C * W];
                    }
                }
            }
        } else {
            FAIL() << "Unsupported precision " << inputPrecision;
        }
    }

    void RunModel(Blob::Ptr& swOutput, Blob::Ptr& hwOutput,
                  const char* model,
                  const char* inputBlobName, const char* outputBlobName,
                  const char* inputFile,
                  bool swapChannels = false,
                  float inputNorm = 1.0f,
                  const std::string& networkConfig = "",
                  const std::string& hwBlackList = "") {
        ASSERT_NO_FATAL_FAILURE(LoadNetwork(model, inputBlobName, outputBlobName));

        _inputInfo->setInputPrecision(inputPrecision);
        _outputInfo->setPrecision(outputPrecision);

        ASSERT_NO_FATAL_FAILURE(LoadInput(inputFile, swapChannels));

        {
            SCOPED_TRACE("SW");
            ASSERT_NO_FATAL_FAILURE(RunNetwork(_net_reader.getNetwork(), false, _input, swOutput,
                                               inputBlobName, outputBlobName,
                                               inputNorm));
        }

        {
            SCOPED_TRACE("HW");
            ASSERT_NO_FATAL_FAILURE(RunNetwork(_net_reader.getNetwork(), true, _input, hwOutput,
                                               inputBlobName, outputBlobName,
                                               inputNorm, networkConfig, hwBlackList));
            ASSERT_NO_FATAL_FAILURE(CheckHWRun());
        }
    }

    Blob::Ptr getFp32Blob(const Blob::Ptr& in) {
        if (in->getTensorDesc().getPrecision() == Precision::FP32)
            return in;

        auto out = make_shared_blob<float, const SizeVector>(Precision::FP32, in->getTensorDesc().getLayout(), in->dims());
        out->allocate();

        if (in->getTensorDesc().getPrecision() == Precision::FP16) {
            PrecisionUtils::f16tof32Arrays(out->buffer().as<float *>(), in->cbuffer().as<ie_fp16 *>(), in->size());
        } else {
            ADD_FAILURE() << "Unsupported precision " << in->getTensorDesc().getPrecision();
        }

        return out;
    }

    Blob::Ptr getFp16Blob(const Blob::Ptr& in) {
        if (in->getTensorDesc().getPrecision() == Precision::FP16)
            return in;

        auto out = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, in->getTensorDesc().getLayout(), in->dims());
        out->allocate();

        if (in->getTensorDesc().getPrecision() == Precision::FP32) {
            PrecisionUtils::f32tof16Arrays(out->buffer().as<ie_fp16 *>(), in->cbuffer().as<float *>(), in->size());
        } else {
            ADD_FAILURE() << "Unsupported precision " << in->getTensorDesc().getPrecision();
        }

        return out;
    }

    void ParseClassifyOutput(const Blob::Ptr& output, std::vector<std::pair<int, float>>& res) {
        res.resize(output->size());

        auto inPtr = output->cbuffer().as<const float*>();
        for (size_t i = 0; i < res.size(); ++i) {
            res[i].first = i;
            res[i].second = inPtr[i];
        }

        std::sort(res.begin(), res.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second > b.second;
        });
    }

    void RunClassifyTest(const char* model,
                         const char* inputBlobName, const char* outputBlobName,
                         const char* inputFile,
                         size_t topK, float topKTolerance,
                         bool swapChannels = false,
                         float inputNorm = 1.0f,
                         const std::string& networkConfig = "",
                         const std::string& hwBlackList = "") {
        if (!CheckMyriadX()) {
            SKIP() << "Non-MyriadX device";
        }

        Blob::Ptr swOutput, hwOutput;
        ASSERT_NO_FATAL_FAILURE(
            RunModel(
                swOutput, hwOutput,
                model,
                inputBlobName, outputBlobName,
                inputFile,
                swapChannels,
                inputNorm,
                networkConfig,
                hwBlackList));

        swOutput = getFp32Blob(swOutput);
        hwOutput = getFp32Blob(hwOutput);

        std::vector<std::pair<int, float>> swRes, hwRes;
        ParseClassifyOutput(swOutput, swRes);
        ParseClassifyOutput(hwOutput, hwRes);

        ASSERT_EQ(swRes.size(), hwRes.size());
        ASSERT_GE(swRes.size(), topK);

        swRes.resize(topK);
        hwRes.resize(topK);

        std::cout << "SW top " << topK << ":" << std::endl;
        for (size_t i = 0; i < topK; ++i) {
            std::cout << i << " : " << swRes[i].first << " : " << swRes[i].second * 100 << "%" << std::endl;
        }

        std::cout << "HW top " << topK << ":" << std::endl;
        for (size_t i = 0; i < topK; ++i) {
            std::cout << i << " : " << hwRes[i].first << " : " << hwRes[i].second * 100 << "%" << std::endl;
        }

        // Compare top K

        for (auto &swElem : swRes) {
            auto hwIt = std::find_if(hwRes.cbegin(), hwRes.cend(), [&swElem](const std::pair<int, float> arg) { return swElem.first == arg.first; });
            ASSERT_TRUE(hwIt != hwRes.end());
            auto hwElem = *hwIt;

            auto probDiff = std::fabs(swElem.second - hwElem.second);
            auto probNorm = std::fabs(swElem.second) + std::numeric_limits<float>::epsilon();
            EXPECT_LE(probDiff / probNorm, topKTolerance)
                << swElem.first << " : " << swElem.second << " vs " << hwElem.second;
        }
    }

    struct box {
        float x, y, w, h;
    };

    void getDetectionBoxes(const float *predictions, float thresh,
                           int ln, int side, int classes,
                           std::vector<std::vector<float>>& probs, std::vector<box>& boxes) {
        boxes.resize(side * side * ln);
        probs.resize(side * side * ln);
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i].resize(classes);
        }

        for (int i = 0; i < side * side; ++i) {
            int row = i / side;
            int col = i % side;
            for (int n = 0; n < ln; ++n) {
                int index = i * ln + n;
                int p_index = side * side * classes + i * ln + n;
                float scale = predictions[p_index];
                int box_index = side * side * (classes + ln) + (i * ln + n) * 4;

                boxes[index].x = (predictions[box_index + 0] + col) / side;
                boxes[index].y = (predictions[box_index + 1] + row) / side;
                boxes[index].w = std::pow(predictions[box_index + 2], 2.0f);
                boxes[index].h = std::pow(predictions[box_index + 3], 2.0f);

                for (int j = 0; j < classes; ++j) {
                    int class_index = i * classes;
                    float prob = scale * predictions[class_index + j];
                    probs[index][j] = (prob > thresh) ? prob : 0.0f;
                }
            }
        }
    }

    struct sortable_bbox {
        int index;
        int cclass;
        std::vector<std::vector<float>> *probs;
    };

    float overlap(float x1, float w1, float x2, float w2) {
        float l1 = x1 - w1 / 2;
        float l2 = x2 - w2 / 2;
        float left = l1 > l2 ? l1 : l2;
        float r1 = x1 + w1 / 2;
        float r2 = x2 + w2 / 2;
        float right = r1 < r2 ? r1 : r2;
        return right - left;
    }

    float box_intersection(box a, box b) {
        float w = overlap(a.x, a.w, b.x, b.w);
        float h = overlap(a.y, a.h, b.y, b.h);
        if (w < 0 || h < 0)
            return 0;
        float area = w * h;
        return area;
    }

    float box_union(box a, box b) {
        float i = box_intersection(a, b);
        float u = a.w * a.h + b.w * b.h - i;
        return u;
    }

    float box_iou(box a, box b) {
        return box_intersection(a, b) / box_union(a, b);
    }

    void doNmsSort(std::vector<box>& boxes, std::vector<std::vector<float>>& probs, int total, int classes, float thresh) {
        std::vector<sortable_bbox> s(total);
        for (int i = 0; i < total; ++i) {
            s[i].index = i;
            s[i].cclass = 0;
            s[i].probs = &probs;
        }

        for (int k = 0; k < classes; ++k) {
            for (int i = 0; i < total; ++i) {
                s[i].cclass = k;
            }

            std::sort(s.begin(), s.end(), [](const sortable_bbox& a, const sortable_bbox& b) {
                auto diff = (*a.probs)[a.index][b.cclass] - (*b.probs)[b.index][b.cclass];
                return diff > 0.0f;
            });

            for (int i = 0; i < total; ++i) {
                if (probs[s[i].index][k] == 0)
                    continue;

                auto a = boxes[s[i].index];
                for (int j = i + 1; j < total; ++j) {
                    auto b = boxes[s[j].index];
                    if (box_iou(a, b) > thresh) {
                        probs[s[j].index][k] = 0;
                    }
                }
            }
        }
    }

    struct bbox {
        int left, right, top, bottom;
        float prob;
        int idx;
    };

    int maxIndex(const std::vector<float>& a, int n) {
        if (n <= 0)
            return -1;

        int max_i = 0;
        float max = a[0];
        for (int i = 1; i < n; ++i) {
            if (a[i] > max) {
                max = a[i];
                max_i = i;
            }
        }

        return max_i;
    }

    void getYoloTopResults(const std::vector<box>& boxes, const std::vector<std::vector<float>>& probs,
                           std::vector<bbox>& out,
                           int imw, int imh, int num, float thresh, int classes) {
        out.clear();

        for (int i = 0; i < num; ++i) {
            auto idxClass = maxIndex(probs[i], classes);
            auto prob = probs[i][idxClass];

            if (prob > thresh) {
                auto b = boxes[i];

                bbox bb;
                bb.left = (b.x - b.w / 2.0f) * imw;
                bb.right = (b.x + b.w / 2.0f) * imw;
                bb.top = (b.y - b.h / 2.0f) * imh;
                bb.bottom = (b.y + b.h / 2.0f) * imh;

                if (bb.left < 0)
                    bb.left = 0;
                if (bb.right > imw - 1)
                    bb.right = imw - 1;
                if (bb.top < 0)
                    bb.top = 0;
                if (bb.bottom > imh - 1)
                    bb.bottom = imh - 1;

                bb.prob = prob;
                bb.idx = idxClass;

                out.push_back(bb);
            }
        }
    }

    void compareBBoxes(const std::vector<bbox>& swRes, const std::vector<bbox>& hwRes,
                       int imw, int imh,
                       float boxTolerance, float probTolerance) {
        std::cout << "SW top: " << std::endl;
        for (size_t i = 0; i < swRes.size(); ++i) {
            const auto& bb = swRes[i];
            std::cout << i << " : " << bb.idx
                      << " : [("
                      << bb.left << " " << bb.top << "), ("
                      << bb.right << " " << bb.bottom
                      << ")] : "
                      << bb.prob * 100 << "%" << std::endl;
        }

        std::cout << "HW top: " << std::endl;
        for (size_t i = 0; i < hwRes.size(); ++i) {
            const auto& bb = hwRes[i];
            std::cout << i << " : " << bb.idx
                      << " : [("
                      << bb.left << " " << bb.top << "), ("
                      << bb.right << " " << bb.bottom
                      << ")] : "
                      << bb.prob * 100 << "%" << std::endl;
        }

        ASSERT_GE(hwRes.size(), swRes.size());
        for (const auto& swBb : swRes) {
            bool found = false;

            float maxBoxError = 0.0f;
            float maxProbError = 0.0f;

            for (const auto& hwBb : hwRes) {
                if (hwBb.idx != swBb.idx)
                    continue;

                box a{
                    static_cast<float>(hwBb.left) / imw,
                    static_cast<float>(hwBb.top) / imh,
                    static_cast<float>(hwBb.right - hwBb.left) / imw,
                    static_cast<float>(hwBb.bottom - hwBb.top) / imh
                };
                box b{
                    static_cast<float>(swBb.left) / imw,
                    static_cast<float>(swBb.top) / imh,
                    static_cast<float>(swBb.right - swBb.left) / imw,
                    static_cast<float>(swBb.bottom - swBb.top) / imh
                };

                auto boxError = box_iou(a, b);
                maxBoxError = std::max(maxBoxError, boxError);

                auto probError = std::fabs(hwBb.prob - swBb.prob);
                maxProbError = std::max(maxProbError, probError);

                if (boxError < boxTolerance) {
                    continue;
                }

                if (probError > probTolerance) {
                    continue;
                }

                found = true;
                break;
            }

            EXPECT_TRUE(found)
                    << "maxBoxError=" << maxBoxError << " "
                    << "maxProbError=" << maxProbError;
        }
    }

    void RunYoloV1Test(const char* model,
                       const char* inputBlobName, const char* outputBlobName,
                       const char* inputFile,
                       int imw, int imh,
                       int ln, int side, int classes,
                       float thresh, float nms,
                       float boxTolerance, float probTolerance,
                       bool swapChannels = false,
                       float inputNorm = 1.0f,
                       const std::string& networkConfig = "",
                       const std::string& hwBlackList = "") {
        if (!CheckMyriadX()) {
            SKIP() << "Non-MyriadX device";
        }

        Blob::Ptr swOutput, hwOutput;
        ASSERT_NO_FATAL_FAILURE(
            RunModel(
                swOutput, hwOutput,
                model,
                inputBlobName, outputBlobName,
                inputFile,
                swapChannels,
                inputNorm,
                networkConfig,
                hwBlackList));

        swOutput = getFp32Blob(swOutput);
        hwOutput = getFp32Blob(hwOutput);

        std::vector<std::vector<float>> swProbs, hwProbs;
        std::vector<box> swBoxes, hwBoxes;
        getDetectionBoxes(swOutput->cbuffer().as<const float*>(), thresh, ln, side, classes, swProbs, swBoxes);
        getDetectionBoxes(hwOutput->cbuffer().as<const float*>(), thresh, ln, side, classes, hwProbs, hwBoxes);

        doNmsSort(swBoxes, swProbs, side * side * ln, classes, nms);
        doNmsSort(hwBoxes, hwProbs, side * side * ln, classes, nms);

        std::vector<bbox> swRes, hwRes;
        getYoloTopResults(swBoxes, swProbs, swRes, imw, imh, side * side * ln, thresh, classes);
        getYoloTopResults(hwBoxes, hwProbs, hwRes, imw, imh, side * side * ln, thresh, classes);

        compareBBoxes(swRes, hwRes, imw, imh, boxTolerance, probTolerance);
    }

    int entryIndex(int lw, int lh, int lcoords, int lclasses, int lnum, int batch, int location, int entry) {
        int n =   location / (lw * lh);
        int loc = location % (lw * lh);
        int loutputs = lh * lw * lnum * (lclasses + lcoords + 1);
        return batch * loutputs + n * lw * lh * (lcoords + lclasses + 1) + entry * lw * lh + loc;
    }

    box getRegionBox(const float* x, const float* biases, int n, int index, int i, int j, int w, int h, int stride) {
        box b;
        b.x = (i + x[index + 0*stride]) / w;
        b.y = (j + x[index + 1*stride]) / h;
        b.w = std::exp(x[index + 2*stride]) * biases[2*n]   / w;
        b.h = std::exp(x[index + 3*stride]) * biases[2*n+1] / h;
        return b;
    }

    void correctRegionBoxes(box* boxes, int n, int w, int h, int netw, int neth, int relative) {
        int new_w=0;
        int new_h=0;
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw)/w;
        } else {
            new_h = neth;
            new_w = (w * neth)/h;
        }

        for (int i = 0; i < n; ++i) {
            box b = boxes[i];
            b.x =  (b.x - (netw - new_w) / 2.0 / netw) / ((float)new_w / netw);
            b.y =  (b.y - (neth - new_h) / 2.0 / neth) / ((float)new_h / neth);
            b.w *= (float)netw / new_w;
            b.h *= (float)neth / new_h;
            if (!relative) {
                b.x *= w;
                b.w *= w;
                b.y *= h;
                b.h *= h;
            }
            boxes[i] = b;
        }
    }

    void getRegionBoxes(const float* predictions, int lw, int lh,
                        int lcoords, int lclasses, int lnum, int w, int h, int netw, int neth,
                        float thresh,
                        std::vector<std::vector<float>>& probs, std::vector<box>& boxes,
                        int relative,
                        const float *anchors) {
        boxes.resize(lw * lh * lnum);
        probs.resize(lw * lh * lnum);
        for (size_t i = 0; i < probs.size(); ++i) {
            probs[i].resize(lclasses + 1);
        }

        for (int i = 0; i < lw * lh; ++i) {
            int row = i / lw;
            int col = i % lw;

            for (int n = 0; n < lnum; ++n) {
                int index = n * lw * lh + i;
                for (int j = 0; j < lclasses; ++j) {
                    probs[index][j] = 0;
                }

                int obj_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords);
                int box_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, 0);
                float scale = predictions[obj_index];

                boxes[index] = getRegionBox(predictions, anchors, n, box_index, col, row, lw, lh, lw * lh);

                int class_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords + 1);

                float max = 0;
                for (int j = 0; j < lclasses; ++j) {
                    int class_index = entryIndex(lw, lh, lcoords, lclasses, lnum, 0, n * lw * lh + i, lcoords + 1 + j);
                    float prob = scale * predictions[class_index];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                    if (prob > max)
                        max = prob;
                }

                probs[index][lclasses] = max;
            }
        }

        correctRegionBoxes(boxes.data(), lw * lh * lnum, w, h, netw, neth, relative);
    }

    void RunYoloV2Test(const char* model,
                       const char* inputBlobName, const char* outputBlobName,
                       const char* inputFile,
                       int imw, int imh,
                       int coords, int num, int classes,
                       int lw, int lh,
                       const float *anchors,
                       float thresh, float nms,
                       float boxTolerance, float probTolerance,
                       bool swapChannels = false,
                       float inputNorm = 1.0f,
                       const std::string& networkConfig = "",
                       const std::string& hwBlackList = "") {
        if (!CheckMyriadX()) {
            SKIP() << "Non-MyriadX device";
        }

        Blob::Ptr swOutput, hwOutput;
        ASSERT_NO_FATAL_FAILURE(
            RunModel(
                swOutput, hwOutput,
                model,
                inputBlobName, outputBlobName,
                inputFile,
                swapChannels,
                inputNorm,
                networkConfig,
                hwBlackList));

        swOutput = getFp32Blob(swOutput);
        hwOutput = getFp32Blob(hwOutput);

        std::vector<std::vector<float>> swProbs, hwProbs;
        std::vector<box> swBoxes, hwBoxes;
        getRegionBoxes(swOutput->cbuffer().as<const float*>(), lw, lh,
                       coords, classes, num, imw, imh, imw, imh,
                       thresh,
                       swProbs, swBoxes,
                       1,
                       anchors);
        getRegionBoxes(hwOutput->cbuffer().as<const float*>(), lw, lh,
                       coords, classes, num, imw, imh, imw, imh,
                       thresh,
                       hwProbs, hwBoxes,
                       1,
                       anchors);

        doNmsSort(swBoxes, swProbs, lw * lh * num, classes, nms);
        doNmsSort(hwBoxes, hwProbs, lw * lh * num, classes, nms);

        std::vector<bbox> swRes, hwRes;
        getYoloTopResults(swBoxes, swProbs, swRes, imw, imh, lw * lh * num, thresh, classes);
        getYoloTopResults(hwBoxes, hwProbs, hwRes, imw, imh, lw * lh * num, thresh, classes);

        compareBBoxes(swRes, hwRes, imw, imh, boxTolerance, probTolerance);
    }

    void parseDetections(const float* ptr, int count, float conf_thresh, int width, int height,
                         std::vector<bbox>& out) {
        out.clear();

        for (int i = 0; i < count; ++i) {
            int batch_id = ptr[i * 7 + 0];
            if (batch_id < 0)
                continue;

            int class_id = ptr[i * 7 + 1];

            float conf = ptr[i * 7 + 2];
            if (conf < conf_thresh)
                continue;

            float xmin = ptr[i * 7 + 3];
            float ymin = ptr[i * 7 + 4];
            float xmax = ptr[i * 7 + 5];
            float ymax = ptr[i * 7 + 6];

            bbox b;
            b.idx = class_id;
            b.prob = conf;
            b.left = width * xmin;
            b.right = width * xmax;
            b.top = height * ymin;
            b.bottom = height * ymax;

            out.push_back(b);
        }
    }

    void RunDetectionTest(const char* model,
                          const char* inputBlobName, const char* outputBlobName,
                          const char* inputFile,
                          int imw, int imh,
                          float conf_thresh,
                          float boxTolerance, float probTolerance,
                          bool swapChannels = false,
                          float inputNorm = 1.0f,
                          const std::string& networkConfig = "",
                          const std::string& hwBlackList = "") {
        if (!CheckMyriadX()) {
            SKIP() << "Non-MyriadX device";
        }

        Blob::Ptr swOutput, hwOutput;
        ASSERT_NO_FATAL_FAILURE(
            RunModel(
                swOutput, hwOutput,
                model,
                inputBlobName, outputBlobName,
                inputFile,
                swapChannels,
                inputNorm,
                networkConfig,
                hwBlackList));

        swOutput = getFp32Blob(swOutput);
        hwOutput = getFp32Blob(hwOutput);

        std::vector<bbox> swRes, hwRes;
        parseDetections(swOutput->cbuffer().as<const float*>(),
                        swOutput->size() / 7,
                        conf_thresh,
                        imw, imh,
                        swRes);
        parseDetections(hwOutput->cbuffer().as<const float*>(),
                        hwOutput->size() / 7,
                        conf_thresh,
                        imw, imh,
                        hwRes);

        compareBBoxes(swRes, hwRes, imw, imh, boxTolerance, probTolerance);
    }

    void RunGenericTest(const char* model,
                        const char* inputBlobName, const char* outputBlobName,
                        const char* inputFile,
                        float tolerance,
                        bool swapChannels = false,
                        float inputNorm = 1.0f,
                        const std::string& networkConfig = "",
                        const std::string& hwBlackList = "") {
        if (!CheckMyriadX()) {
            SKIP() << "Non-MyriadX device";
        }

        Blob::Ptr swOutput, hwOutput;
        ASSERT_NO_FATAL_FAILURE(
            RunModel(
                swOutput, hwOutput,
                model,
                inputBlobName, outputBlobName,
                inputFile,
                swapChannels,
                inputNorm,
                networkConfig,
                hwBlackList));

        swOutput = getFp16Blob(swOutput);
        hwOutput = getFp16Blob(hwOutput);

        Compare(hwOutput, swOutput, tolerance);
    }

    void RunAsyncTest(const char* model,
                      const char* inputBlobName, const char* outputBlobName,
                      const char* inputFile,
                      int numIters = 20,
                      bool swapChannels = false,
                      float inputNorm = 1.0f,
                      const std::string& networkConfig = "",
                      const std::string& hwBlackList = "") {
        if (!CheckMyriadX()) {
            SKIP() << "Non-MyriadX device";
        }

        const int NUM_REQUESTS = 4;

        ASSERT_NO_FATAL_FAILURE(LoadNetwork(model, inputBlobName, outputBlobName));

        _inputInfo->setInputPrecision(inputPrecision);
        _outputInfo->setPrecision(outputPrecision);

        ASSERT_NO_FATAL_FAILURE(LoadInput(inputFile, swapChannels));

        StatusCode st;

        ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, _net_reader.getNetwork(),
                                                          {
                                                              {
                                                                  VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION),
                                                                  CONFIG_VALUE(YES)
                                                              },
                                                              {
                                                                  VPU_CONFIG_KEY(INPUT_NORM),
                                                                  std::to_string(inputNorm)
                                                              },
                                                              {
                                                                  VPU_CONFIG_KEY(NETWORK_CONFIG),
                                                                  networkConfig
                                                              },
                                                              {
                                                                  VPU_CONFIG_KEY(HW_BLACK_LIST),
                                                                  hwBlackList
                                                              },
                                                              {
                                                                  CONFIG_KEY(PERF_COUNT),
                                                                  CONFIG_VALUE(YES)
                                                              }
                                                          },
                                                          &_resp));
        ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

        IInferRequest::Ptr inferRequests[NUM_REQUESTS];
        Blob::Ptr outputs[NUM_REQUESTS];

        for (int i = 0; i < NUM_REQUESTS; ++i) {
            ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(inferRequests[i], &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

            ASSERT_NO_THROW(st = inferRequests[i]->SetBlob(inputBlobName, _input, &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

            ASSERT_NO_THROW(st = inferRequests[i]->GetBlob(outputBlobName, outputs[i], &_resp));
            ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
        }

        std::vector<Blob::Ptr> allOutputs[NUM_REQUESTS];
        for (int i = 0; i < NUM_REQUESTS; ++i) {
            allOutputs[i].resize(numIters);
        }

        for (int iterInd = 0; iterInd < numIters; ++iterInd) {
            for (int inferInd = 0; inferInd < NUM_REQUESTS; ++inferInd) {
                ASSERT_NO_THROW(st = inferRequests[inferInd]->StartAsync(&_resp));
                ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
            }

            for (int inferInd = 0; inferInd < NUM_REQUESTS; ++inferInd) {
                ASSERT_NO_THROW(st = inferRequests[inferInd]->Wait(IInferRequest::RESULT_READY, &_resp));
                ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
            }

            for (int inferInd = 0; inferInd < NUM_REQUESTS; ++inferInd) {

                allOutputs[inferInd][iterInd] = make_blob_with_precision(Precision::FP16, outputs[inferInd]->layout(), outputs[inferInd]->dims());
                allOutputs[inferInd][iterInd]->allocate();

                auto outputFP16 = getFp16Blob(outputs[inferInd]);

                memcpy(allOutputs[inferInd][iterInd]->buffer(), outputFP16->cbuffer(), outputFP16->byteSize());
            }
        }

        for (int iterInd1 = 0; iterInd1 < numIters; ++iterInd1) {
            for (int iterInd2 = iterInd1; iterInd2 < numIters; ++iterInd2) {
                for (int inferInd1 = 0; inferInd1 < NUM_REQUESTS; ++inferInd1) {
                    for (int inferInd2 = inferInd1; inferInd2 < NUM_REQUESTS; ++inferInd2) {
                        ASSERT_NO_FATAL_FAILURE(Compare(allOutputs[inferInd1][iterInd1], allOutputs[inferInd2][iterInd2], 0.0f))
                                << "inferInd1=" << inferInd1 << " "
                                << "iterInd1=" << iterInd1 << " "
                                << "inferInd2=" << inferInd2 << " "
                                << "iterInd2=" << iterInd2;
                    }
                }
            }
        }
    }
};

// GoogleNetV1

TEST_P(myriadXHWNetworks_nightly, GoogleNetV1_Single) {
    RunClassifyTest("googlenet/bvlc_googlenet_fp16",
                    "data", "prob",
                    "224x224/cat3.bmp",
                    7, 0.12f);
}

TEST_P(myriadXHWNetworks_nightly, GoogleNetV1_Async) {
    RunAsyncTest("googlenet/bvlc_googlenet_fp16",
                 "data", "prob",
                 "224x224/cat3.bmp",
                 100);
}

// BNGoogleNetV1 (old)

TEST_P(myriadXHWNetworks_nightly, BNGoogleNetV1_Old_Single) {
    RunClassifyTest("BNGoogleNet/BNGoogleNet_fp16",
                    "input", "loss3_classifier",
                    "224x224/cat3.bmp",
                    7, 0.03f,
                    false, 1.0f,
                    "data=input,scale=1");
}

TEST_P(myriadXHWNetworks_nightly, BNGoogleNetV1_Old_Async) {
    RunAsyncTest("BNGoogleNet/BNGoogleNet_fp16",
                 "input", "loss3_classifier",
                 "224x224/cat3.bmp",
                 100, false,
                 1.0f,
                 "data=input,scale=1");
}

// BNGoogleNetV1_HorizontalFusion

TEST_P(myriadXHWNetworks_nightly, BNGoogleNetV1_HorizontalFusion_Single) {
    RunClassifyTest("BNGoogleNet_Horizontal_fusion/BNGoogleNetHorizontalFusion_fp16",
                    "data", "loss3/classifier",
                    "224x224/cat3.bmp",
                    7, 0.03f,
                    false, 1.0f,
                    "data=data,scale=1");
}

TEST_P(myriadXHWNetworks_nightly, BNGoogleNetV1_HorizontalFusion_Async) {
    RunAsyncTest("BNGoogleNet_Horizontal_fusion/BNGoogleNetHorizontalFusion_fp16",
                 "data", "loss3/classifier",
                 "224x224/cat3.bmp",
                 100, false,
                 1.0f,
                 "data=data,scale=1");
}

// BNGoogleNetV1 (new)

TEST_P(myriadXHWNetworks_nightly, BNGoogleNetV1_New_Single) {
    RunClassifyTest("BNGoogleNet/BNGoogleNet_new_fp16",
                    "data", "loss3/classifier",
                    "224x224/cat3.bmp",
                    7, 0.03f,
                    false, 1.0f,
                    "data=data,scale=1");
}

TEST_P(myriadXHWNetworks_nightly, BNGoogleNetV1_New_Async) {
    RunAsyncTest("BNGoogleNet/BNGoogleNet_new_fp16",
                 "data", "loss3/classifier",
                 "224x224/cat3.bmp",
                 100, false,
                 1.0f,
                 "data=data,scale=1");
}

// GoogleNetV2

TEST_P(myriadXHWNetworks_nightly, GoogleNetV2_Single) {
    RunClassifyTest("GoogleNet-V2/CommunityGoogleNetV2_fp16",
                    "input", "prob",
                    "224x224/cat3.bmp",
                    5, 0.555f,
                    false, 1.0f);
}

TEST_P(myriadXHWNetworks_nightly, GoogleNetV2_Async) {
    RunAsyncTest("GoogleNet-V2/CommunityGoogleNetV2_fp16",
                 "input", "prob",
                 "224x224/cat3.bmp",
                 100,
                 false, 1.0f);
}

// YoloTinyV1

TEST_P(myriadXHWNetworks_nightly, YoloTinyV1_Single) {
    RunYoloV1Test("vpu_validation/yolo_tiny_v1_model_ref_fp16",
                  "data", "fc9",
                  "448x448/dog_croped448.bmp",
                  448, 448,
                  2, 7, 20,
                  0.25f, 0.4f,
                  0.75f, 0.24f,
                  false, 1.0f);
}

TEST_P(myriadXHWNetworks_nightly, YoloTinyV1_Async) {
    RunAsyncTest("vpu_validation/yolo_tiny_v1_model_ref_fp16",
                 "data", "fc9",
                 "448x448/dog_croped448.bmp",
                 100,
                 false, 1.0f);
}

// HWYoloTinyV1_FC

TEST_P(myriadXHWNetworks_nightly, HWYoloTinyV1_FC_Single) {
    RunGenericTest("vpu_validation/tiny-yolo-v1-simple-relu-fc",
                   "data", "fc9",
                   "448x448/dog_croped448.bmp",
                   0.022f,
                   false, 1.0f,
                   "data=data,scale=256");
}

TEST_P(myriadXHWNetworks_nightly, HWYoloTinyV1_FC_Async) {
    RunAsyncTest("vpu_validation/tiny-yolo-v1-simple-relu-fc",
                 "data", "fc9",
                 "448x448/dog_croped448.bmp",
                 100,
                 false, 1.0f);
}

// HWYoloTinyV1_Conv

TEST_P(myriadXHWNetworks_nightly, HWYoloTinyV1_Conv_Single) {
    RunGenericTest("vpu_validation/tiny-yolo-v1-simple-relu-conv",
                   "data", "fc9",
                   "448x448/dog_croped448.bmp",
                   10.0f);
}

TEST_P(myriadXHWNetworks_nightly, HWYoloTinyV1_Conv_Async) {
    RunAsyncTest("vpu_validation/tiny-yolo-v1-simple-relu-conv",
                 "data", "fc9",
                 "448x448/dog_croped448.bmp",
                 100);
}

// ResNet-18

TEST_P(myriadXHWNetworks_nightly, ResNet18_Single) {
    RunClassifyTest("resnet-18/ResNet-18_fp16",
                    "data", "loss",
                    "224x224/cat3.bmp",
                    2, 0.266f,
                    false, 1.0f);
}

TEST_P(myriadXHWNetworks_nightly, ResNet18_Async) {
    RunAsyncTest("resnet-18/ResNet-18_fp16",
                 "data", "loss",
                 "224x224/cat3.bmp",
                 100,
                 false, 1.0f);
}

// ReID Embedding

TEST_P(myriadXHWNetworks_nightly, ReID_Embedding_Single) {
    RunGenericTest("vpu_validation/reid_embedding_fp16",
                   "input", "conv3_4",
                   "vpu/embedding.bmp",
                   2.6f, false,
                   256.0);
}

TEST_P(myriadXHWNetworks_nightly, ReID_Embedding_Async) {
    RunAsyncTest("vpu_validation/reid_embedding_fp16",
                 "input", "conv3_4",
                 "vpu/embedding.bmp",
                 100);
}

// ResNet-50

TEST_P(myriadXHWNetworks_nightly, ResNet50_Single) {
    RunClassifyTest("ResNet-50/ResNet-50_fp16",
                    "data", "prob",
                    "224x224/cat3.bmp",
                    3, 2.04f);
}

TEST_P(myriadXHWNetworks_nightly, ResNet50_Async) {
    RunAsyncTest("ResNet-50/ResNet-50_fp16",
                 "data", "prob",
                 "224x224/cat3.bmp",
                 100);
}

// SqueezeNet10

TEST_P(myriadXHWNetworks_nightly, SqueezeNet10_Single) {
    RunClassifyTest("squeezenet_1.0/SqueezeNet_1_0_fp16_no_mean",
                    "data", "prob",
                    "227x227/cat3.bmp",
                    5, 0.1f);
}

TEST_P(myriadXHWNetworks_nightly, SqueezeNet10_Async) {
    RunAsyncTest("squeezenet_1.0/SqueezeNet_1_0_fp16_no_mean",
                 "data", "prob",
                 "227x227/cat3.bmp",
                 100);
}

// SqueezeNet11

TEST_P(myriadXHWNetworks_nightly, SqueezeNet11_Single) {
    RunClassifyTest("SqueezeNet_v1.1/SqueezeNet_v1.1_fp16_no_mean",
                    "input", "prob",
                    "227x227/cat3.bmp",
                    5, 0.252f);
}

TEST_P(myriadXHWNetworks_nightly, SqueezeNet11_Async) {
    RunAsyncTest("SqueezeNet_v1.1/SqueezeNet_v1.1_fp16_no_mean",
                 "input", "prob",
                 "227x227/cat3.bmp",
                 100);
}

// VGG16

TEST_P(myriadXHWNetworks_nightly, VGG16_Single) {
    DISABLE_ON_ARM;
    RunClassifyTest("vgg/VGG_ILSVRC_16_layers_fp16_no_mean",
                    "input", "prob",
                    "224x224/cat3.bmp",
                    5, 0.41f,
                    false, 1.0f,
                    "data=input,scale=1");
}

TEST_P(myriadXHWNetworks_nightly, VGG16_Async) {
    DISABLE_ON_ARM;
    RunAsyncTest("vgg/VGG_ILSVRC_16_layers_fp16_no_mean",
                 "input", "prob",
                 "224x224/cat3.bmp",
                 20, false,
                 1.0f,
                 "data=input,scale=1");
}

// VGG16 TensorFlow

TEST_P(myriadXHWNetworks_nightly, VGG16_TF_Single) {
    DISABLE_ON_ARM;
    RunClassifyTest("vgg-16/vgg16_tf_fp16",
                    "input", "vgg_16/fc8/convolution",
                    "224x224/cat3.bmp",
                    5, 0.41f,
                    false, 1.0f,
                    "data=input,scale=1");
}

TEST_P(myriadXHWNetworks_nightly, VGG16_TF_Async) {
    DISABLE_ON_ARM;
    RunAsyncTest("vgg-16/vgg16_tf_fp16",
                 "input", "vgg_16/fc8/convolution",
                 "224x224/cat3.bmp",
                 20, false,
                 1.0f,
                 "data=input,scale=1");
}

// VGG19

TEST_P(myriadXHWNetworks_nightly, VGG19_Single) {
    DISABLE_ON_ARM;
    RunClassifyTest("vgg-19/VGG_ILSVRC_19_layers_fp16_no_mean",
                    "input", "prob",
                    "224x224/cat3.bmp",
                    5, 0.4f,
                    false, 1.0f,
                    "data=input,scale=1");
}

TEST_P(myriadXHWNetworks_nightly, VGG19_Async) {
    DISABLE_ON_ARM;
    RunAsyncTest("vgg-19/VGG_ILSVRC_19_layers_fp16_no_mean",
                 "input", "prob",
                 "224x224/cat3.bmp",
                 20, false,
                 1.0f,
                 "data=input,scale=1");
}

// MobileNet-SSD
// TODO : investigate failure on HDDL plugin

TEST_P(myriadXHWNetworks_nightly, MobileNet_SSD_Single) {
#ifdef USE_HDDL
    if (std::getenv("IE_VPU_ENABLE_PER_LAYER_TESTS_HDDL")) {
        SKIP() << "HDDL bug";
    }
#endif

    std::string confFilePath = ModelsPath() + "/MobileNet-SSD/MobileNet-SSD_fp16.conf.xml";

    RunDetectionTest("MobileNet-SSD/MobileNet-SSD_fp16",
                     "input", "detection_out",
                     "300x300/dog.bmp",
                     300, 300,
                     0.3f,
                     0.85f, 0.1f,
                     false, 1.0f,
                     "file=" + confFilePath);
}

TEST_P(myriadXHWNetworks_nightly, MobileNet_SSD_Async) {
#ifdef USE_HDDL
    if (std::getenv("IE_VPU_ENABLE_PER_LAYER_TESTS_HDDL")) {
        SKIP() << "HDDL bug";
    }
#endif

    std::string confFilePath = ModelsPath() + "/MobileNet-SSD/MobileNet-SSD_fp16.conf.xml";

    RunAsyncTest("MobileNet-SSD/MobileNet-SSD_fp16",
                 "input", "detection_out",
                 "300x300/dog.bmp",
                 20,
                 false, 1.0f,
                 "file=" + confFilePath);
}

// YoloTinyV2_Caffe

TEST_P(myriadXHWNetworks_nightly, YoloTinyV2_Caffe_Single) {
    const float TINY_YOLOV2_ANCHORS[] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};

    RunYoloV2Test("yolo_tiny_v2/TinyYoloV2_fp16",
                  "input", "Region",
                  "416x416/person.bmp",
                  416, 416,
                  4, 5, 20,
                  13, 13,
                  TINY_YOLOV2_ANCHORS,
                  0.25f, 0.4f,
                  0.9f, 0.015f,
                  true, 255.0f,
                  "data=input,scale=256");
}

TEST_P(myriadXHWNetworks_nightly, YoloTinyV2_Caffe_Async) {
    RunAsyncTest("yolo_tiny_v2/TinyYoloV2_fp16",
                 "input", "Region",
                 "416x416/person.bmp",
                 40,
                 true, 255.0f,
                 "data=input,scale=256");
}

// YoloTinyV2_DarkNet
// TODO: FIXME, CVS-11259

TEST_P(myriadXHWNetworks_nightly, YoloTinyV2_DarkNet_Single) {
    RunGenericTest("vpu_validation/yolo_tiny_v2_model_ref",
                   "input", "22-convolutional",
                   "416x416/person.bmp",
                   0.22f,
                   false, 1.0f,
                   "data=input,scale=256");
}

TEST_P(myriadXHWNetworks_nightly, YoloTinyV2_DarkNet_Async) {
    RunAsyncTest("vpu_validation/yolo_tiny_v2_model_ref",
                 "input", "22-convolutional",
                 "416x416/person.bmp",
                 100,
                 false, 1.0f,
                 "data=input,scale=256");
}

// AgeGender

TEST_P(myriadXHWNetworks_nightly, AgeGender_Single) {
    RunGenericTest("vpu_validation/age_gender_net_model_ref",
                   "data", "prob",
                   "62x62/face62.bmp",
                   0.005f,
                   false, 1.0f,
                   "data=data,scale=128");

    RunGenericTest("vpu_validation/age_gender_net_model_ref",
                   "data", "age_conv3",
                   "62x62/face62.bmp",
                   0.005f,
                   false, 1.0f,
                   "data=data,scale=128");
}

TEST_P(myriadXHWNetworks_nightly, AgeGender_Async) {
    RunAsyncTest("vpu_validation/age_gender_net_model_ref",
                 "data", "prob",
                 "62x62/face62.bmp",
                 100,
                 false, 1.0f,
                 "data=data,scale=64");

    RunAsyncTest("vpu_validation/age_gender_net_model_ref",
                 "data", "age_conv3",
                 "62x62/face62.bmp",
                 100,
                 false, 1.0f,
                 "data=data,scale=64");
}

// AlexNet

TEST_P(myriadXHWNetworks_nightly, AlexNet_Single) {
    RunClassifyTest("alexnet/bvlc_alexnet_fp16_mf",
                    "data", "prob",
                    "227x227/cat3.bmp",
                    1, 0.17f);
}

TEST_P(myriadXHWNetworks_nightly, AlexNet_Async) {
    RunAsyncTest("alexnet/bvlc_alexnet_fp16_mf",
                 "data", "prob",
                 "227x227/cat3.bmp",
                 100);
}

std::string getTestCaseName(const testing::TestParamInfo<HwNetworkParams>& param) {
    return std::string((std::get<0>(param.param)).name()) + "_" +
           std::string((std::get<1>(param.param)).name());
}

INSTANTIATE_TEST_CASE_P(Input_Output_ExecMode, myriadXHWNetworks_nightly,
        testing::Values(
              std::make_tuple(Precision::FP16, Precision::FP16)
            , std::make_tuple(Precision::U8, Precision::FP32)
        ),
    getTestCaseName
);

//
// SeveralLayers
//

TEST_F(myriadXHWLayersTests_nightly, SeveralLayers) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    tensor_test_params dims1{1, 3, 224, 224};
    tensor_test_params dims2{1, 64, 112, 112};
    tensor_test_params dims3{1, 64, 56, 56};

    param_size kernel1{7, 7};
    param_size stride1{2, 2};
    param_size pad1{3, 3};

    param_size kernel2{3, 3};
    param_size stride2{2, 2};
    param_size pad2{0, 0};

    IN_OUT_desc tensor1, tensor2, tensor3;
    tensor1.push_back({dims1.n, dims1.c, dims1.h, dims1.w});
    tensor2.push_back({dims2.n, dims2.c, dims2.h, dims2.w});
    tensor3.push_back({dims3.n, dims3.c, dims3.h, dims3.w});

    size_t numWeights = kernel1.x * kernel1.y * dims1.c * dims2.c;
    size_t numBiases = dims2.c;

    ParamsStruct convParams = {
              {"kernel-x", std::to_string(kernel1.x)}
            , {"kernel-y", std::to_string(kernel1.y)}
            , {"stride-x", std::to_string(stride1.x)}
            , {"stride-y", std::to_string(stride1.y)}
            , {"pad-x", std::to_string(pad1.x)}
            , {"pad-y", std::to_string(pad1.y)}
            , {"output", std::to_string(dims2.c)}
            , {"group", "1"}
    };
    AddLayer("Convolution",
             &convParams,
             numWeights,
             numBiases,
             defaultWeightsRange,
             tensor1,
             tensor2,
             ref_convolution_wrap);

    ParamsStruct reluParams = {
        {"negative_slope", "0.0"}
    };
    AddLayer("ReLU",
             &reluParams,
             tensor2,
             tensor2,
             ref_ReLU_wrap);

    ParamsStruct poolParams = {
              {"kernel-x", std::to_string(kernel2.x)}
            , {"kernel-y", std::to_string(kernel2.y)}
            , {"stride-x", std::to_string(stride2.x)}
            , {"stride-y", std::to_string(stride2.y)}
            , {"pad-x", std::to_string(pad2.x)}
            , {"pad-y", std::to_string(pad2.y)}
            , {"pool-method", "max"}
    };
    AddLayer("Pooling",
             &poolParams,
             tensor2,
             tensor3,
             ref_pooling_wrap);

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    Compare(hwOutput, swOutput, 0.1f);
}

//
// LargePoolWithConv
//

TEST_F(myriadXHWLayersTests_nightly, LargePoolWithConv) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    tensor_test_params dims1{1, 16, 448, 448};
    tensor_test_params dims2{1, 16, 224, 224};
    tensor_test_params dims3{1, 32, 224, 224};

    param_size kernel1{2, 2};
    param_size stride1{2, 2};
    param_size pad1{0, 0};

    param_size kernel2{3, 3};
    param_size stride2{1, 1};
    param_size pad2{1, 1};

    IN_OUT_desc tensor1, tensor2, tensor3;
    tensor1.push_back({dims1.n, dims1.c, dims1.h, dims1.w});
    tensor2.push_back({dims2.n, dims2.c, dims2.h, dims2.w});
    tensor3.push_back({dims3.n, dims3.c, dims3.h, dims3.w});

    ParamsStruct poolParams = {
              {"kernel-x", std::to_string(kernel1.x)}
            , {"kernel-y", std::to_string(kernel1.y)}
            , {"stride-x", std::to_string(stride1.x)}
            , {"stride-y", std::to_string(stride1.y)}
            , {"pad-x", std::to_string(pad1.x)}
            , {"pad-y", std::to_string(pad1.y)}
            , {"pool-method", "max"}
    };
    AddLayer("Pooling",
             &poolParams,
             tensor1,
             tensor2,
             ref_pooling_wrap);

    size_t numWeights = kernel2.x * kernel2.y * dims2.c * dims3.c;
    size_t numBiases = dims3.c;

    ParamsStruct convParams = {
              {"kernel-x", std::to_string(kernel2.x)}
            , {"kernel-y", std::to_string(kernel2.y)}
            , {"stride-x", std::to_string(stride2.x)}
            , {"stride-y", std::to_string(stride2.y)}
            , {"pad-x", std::to_string(pad2.x)}
            , {"pad-y", std::to_string(pad2.y)}
            , {"output", std::to_string(dims3.c)}
            , {"group", "1"}
    };
    AddLayer("Convolution",
             &convParams,
             numWeights,
             numBiases,
             defaultWeightsRange,
             tensor2,
             tensor3,
             ref_convolution_wrap);

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    Compare(hwOutput, swOutput, 0.08f);
}

TEST_F(myriadXHWLayersTests_nightly, ConvWithPool) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    tensor_test_params dims1{1, 16, 4, 4};
    tensor_test_params dims2{1, 64, 4, 4};
    tensor_test_params dims3{1, 64, 2, 2};

    param_size kernel1{3, 3};
    param_size stride1{1, 1};
    param_size pad1{1, 1};

    param_size kernel2{2, 2};
    param_size stride2{2, 2};
    param_size pad2{0, 0};

    IN_OUT_desc tensor1, tensor2, tensor3;
    tensor1.push_back({dims1.n, dims1.c, dims1.h, dims1.w});
    tensor2.push_back({dims2.n, dims2.c, dims2.h, dims2.w});
    tensor3.push_back({dims3.n, dims3.c, dims3.h, dims3.w});

    size_t numWeights = kernel1.x * kernel1.y * dims1.c * dims2.c;
    size_t numBiases = dims2.c;

    ParamsStruct convParams = {
              {"kernel-x", std::to_string(kernel1.x)}
            , {"kernel-y", std::to_string(kernel1.y)}
            , {"stride-x", std::to_string(stride1.x)}
            , {"stride-y", std::to_string(stride1.y)}
            , {"pad-x", std::to_string(pad1.x)}
            , {"pad-y", std::to_string(pad1.y)}
            , {"output", std::to_string(dims2.c)}
            , {"group", "1"}
    };

    AddLayer("Convolution",
             &convParams,
             numWeights,
             numBiases,
             defaultWeightsRange,
             tensor1,
             tensor2,
             ref_convolution_wrap);

    ParamsStruct poolParams = {
              {"kernel-x", std::to_string(kernel2.x)}
            , {"kernel-y", std::to_string(kernel2.y)}
            , {"stride-x", std::to_string(stride2.x)}
            , {"stride-y", std::to_string(stride2.y)}
            , {"pad-x", std::to_string(pad2.x)}
            , {"pad-y", std::to_string(pad2.y)}
            , {"pool-method", "max"}
    };

    AddLayer("Pooling",
             &poolParams,
             tensor2,
             tensor3,
             ref_pooling_wrap);

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    Compare(hwOutput, swOutput, 0.08f);
}

//
// With concat
//

TEST_F(myriadXHWLayersTests_nightly, WithConcat) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithConcat" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv1" type="Convolution" precision="FP16" id="2">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="16" group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="0" size="512"/>
                    <biases offset="512" size="32"/>
                </layer>
                <layer name="conv2" type="Convolution" precision="FP16" id="3">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="16" group="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="0" size="512"/>
                    <biases offset="512" size="32"/>
                </layer>
                <layer name="conv3" type="Convolution" precision="FP16" id="4">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="16" group="1"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="0" size="512"/>
                    <biases offset="512" size="32"/>
                </layer>
                <layer name="concat" type="Concat" precision="FP16" id="5">
                    <data axis="1"/>
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                        <port id="9">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                        <port id="10">
                            <dim>1</dim>
                            <dim>16</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>48</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                </layer>
                <layer name="last" type="Convolution" precision="FP16" id="6">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="48" group="1"/>
                    <input>
                        <port id="12">
                            <dim>1</dim>
                            <dim>48</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="13">
                            <dim>1</dim>
                            <dim>48</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="544" size="4608"/>
                    <biases offset="5152" size="96"/>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="4"/>
                <edge from-layer="1" from-port="1" to-layer="4" to-port="6"/>
                <edge from-layer="2" from-port="3" to-layer="5" to-port="8"/>
                <edge from-layer="3" from-port="5" to-layer="5" to-port="9"/>
                <edge from-layer="4" from-port="7" to-layer="5" to-port="10"/>
                <edge from-layer="5" from-port="11" to-layer="6" to-port="12"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(5248 / sizeof(ie_fp16)));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.c_str(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weights));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo = _outputsInfo["last"];
    outputInfo->setPrecision(Precision::FP16);

    Blob::Ptr input = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, inputInfo->getDims());
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput, hwOutput;
    {
        SCOPED_TRACE("SW");
        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, false, input, swOutput, "input", "last"));
    }

    {
        SCOPED_TRACE("HW");
        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, true, input, hwOutput, "input", "last"));
        ASSERT_NO_FATAL_FAILURE(CheckHWRun());
    }

    Compare(hwOutput, swOutput, 0.2f);
}

//
// With concat misaligned
//

TEST_F(myriadXHWLayersTests_nightly, WithConcatMisaligned) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithConcat" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                </layer>
                <layer name="conv1" type="Convolution" precision="FP16" id="2">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="35" group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="0" size="2450"/>
                    <biases offset="2450" size="70"/>
                </layer>
                <layer name="conv2" type="Convolution" precision="FP16" id="3">
                    <data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="35" group="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                    <weights offset="0" size="2450"/>
                    <biases offset="2450" size="70"/>
                </layer>
                <layer name="concat" type="Concat" precision="FP16" id="4">
                    <data axis="1"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                        <port id="7">
                            <dim>1</dim>
                            <dim>35</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="8">
                            <dim>1</dim>
                            <dim>70</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                </layer>
                <layer name="last" type="Power" precision="FP16" id="5">
                    <data power="1.0" scale="1.0" shift="0.0"/>
                    <input>
                        <port id="9">
                            <dim>1</dim>
                            <dim>70</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </input>
                    <output>
                        <port id="10">
                            <dim>1</dim>
                            <dim>70</dim>
                            <dim>28</dim>
                            <dim>28</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="4"/>
                <edge from-layer="2" from-port="3" to-layer="4" to-port="6"/>
                <edge from-layer="3" from-port="5" to-layer="4" to-port="7"/>
                <edge from-layer="4" from-port="8" to-layer="5" to-port="9"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(2520 / sizeof(ie_fp16)));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.c_str(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weights));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo = _outputsInfo["last"];
    outputInfo->setPrecision(Precision::FP16);

    Blob::Ptr input = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, inputInfo->getDims());
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput, hwOutput;
    {
        SCOPED_TRACE("SW");
        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, false, input, swOutput, "input", "last"));
    }

    {
        SCOPED_TRACE("HW");
        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, true, input, hwOutput, "input", "last"));
        ASSERT_NO_FATAL_FAILURE(CheckHWRun());
    }

    Compare(hwOutput, swOutput, 0.03f);
}


TEST_F(myriadXHWLayersTests_nightly, With_3_FC_Layers) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithConcat" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>2</dim>
                            <dim>2</dim>
                        </port>
                    </output>
                </layer>
                <layer id="2" name="angle_y" precision="FP16" type="FullyConnected">
                    <data out-size="1"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>2</dim>
                            <dim>2</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                    <blobs>
                        <weights offset="0" size="1024"/>
                        <biases offset="1024" size="2"/>
                    </blobs>
                </layer>
                <layer id="3" name="angle_p" precision="FP16" type="FullyConnected">
                    <data out-size="1"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>2</dim>
                            <dim>2</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                    <blobs>
                        <weights offset="0" size="1024"/>
                        <biases offset="1024" size="2"/>
                    </blobs>
                </layer>
                <layer id="4" name="angle_q" precision="FP16" type="FullyConnected">
                    <data out-size="1"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>128</dim>
                            <dim>2</dim>
                            <dim>2</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>1</dim>
                        </port>
                    </output>
                    <blobs>
                        <weights offset="0" size="1024"/>
                        <biases offset="1024" size="2"/>
                    </blobs>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="0"/>
                <edge from-layer="1" from-port="1" to-layer="4" to-port="0"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights((1024 + 2) / sizeof(ie_fp16)));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.c_str(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weights));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    const std::string names[] = { "angle_p", "angle_q", "angle_y" };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        auto outputInfo = _outputsInfo[names[i]];
        ASSERT_NE(outputInfo, nullptr);
        outputInfo->setPrecision(Precision::FP32);

    }

    Blob::Ptr input = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, inputInfo->getDims());
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput, hwOutput;
    _inferRequest.reset();
    _exeNetwork.reset();

    StatusCode st;

    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network,
                                                          {
                                                          {
                                                              VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)
                                                          },
                                                      },
                                                      &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->SetBlob("input", input, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));

    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    std::vector<float> results(sizeof(names) / sizeof(names[0]));
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        ASSERT_NO_THROW(st = _inferRequest->GetBlob(names[i].c_str(), hwOutput, &_resp));
        ASSERT_NE(hwOutput, nullptr);
        BufferWrapper res_ptr(hwOutput);
        results[i] = res_ptr[0];
    }
    for (size_t i = 1; i < results.size(); ++i) {
        ASSERT_NEAR(results[0], results[i], 0.0001f);
    }
}

//
// With Eltwise
//

TEST_F(myriadXHWLayersTests_nightly, WithEltwise) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithEltwise" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="branch1" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="0" pad-y="0"
                        kernel-x="1" kernel-y="1"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="0" size="8192"/>
                    <biases offset="8192" size="128"/>
                </layer>
                <layer name="branch2a" type="Convolution" precision="FP16" id="3">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="1" pad-y="1"
                        kernel-x="3" kernel-y="3"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="8320" size="73728"/>
                    <biases offset="82048" size="128"/>
                </layer>
                <layer name="branch2a_relu" type="ReLU" precision="FP16" id="4">
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="branch2b" type="Convolution" precision="FP16" id="5">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="1" pad-y="1"
                        kernel-x="3" kernel-y="3"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="82176" size="73728"/>
                    <biases offset="155904" size="128"/>
                </layer>
                <layer name="sum" type="Eltwise" precision="FP16" id="6">
                    <elementwise_data operation="sum"/>
                    <input>
                        <port id="10">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                        <port id="11">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="12">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="last" type="Convolution" precision="FP16" id="7">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="0" pad-y="0"
                        kernel-x="1" kernel-y="1"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="13">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="14">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="156032" size="8192"/>
                    <biases offset="164224" size="128"/>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="4"/>
                <edge from-layer="3" from-port="5" to-layer="4" to-port="6"/>
                <edge from-layer="4" from-port="7" to-layer="5" to-port="8"/>
                <edge from-layer="2" from-port="3" to-layer="6" to-port="10"/>
                <edge from-layer="5" from-port="9" to-layer="6" to-port="11"/>
                <edge from-layer="6" from-port="12" to-layer="7" to-port="13"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(164352 / sizeof(ie_fp16)));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.c_str(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weights));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo = _outputsInfo["last"];
    outputInfo->setPrecision(Precision::FP16);

    Blob::Ptr input = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, inputInfo->getDims());
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput;
    {
        SCOPED_TRACE("SW");
        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, false, input, swOutput, "input", "last"));
    }

    Blob::Ptr hwOutput;
    {
        SCOPED_TRACE("HW");
        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, true, input, hwOutput, "input", "last"));
        ASSERT_NO_FATAL_FAILURE(CheckHWRun());
    }

    Compare(hwOutput, swOutput, 30);
}

TEST_F(myriadXHWLayersTests_nightly, WithEltwiseReLU) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithEltwise" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="branch1" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="0" pad-y="0"
                        kernel-x="1" kernel-y="1"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="0" size="8192"/>
                    <biases offset="8192" size="128"/>
                </layer>
                <layer name="branch2a" type="Convolution" precision="FP16" id="3">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="1" pad-y="1"
                        kernel-x="3" kernel-y="3"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="8320" size="73728"/>
                    <biases offset="82048" size="128"/>
                </layer>
                <layer name="branch2a_relu" type="ReLU" precision="FP16" id="4">
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="branch2b" type="Convolution" precision="FP16" id="5">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="1" pad-y="1"
                        kernel-x="3" kernel-y="3"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="82176" size="73728"/>
                    <biases offset="155904" size="128"/>
                </layer>
                <layer name="sum" type="Eltwise" precision="FP16" id="6">
                    <elementwise_data operation="sum"/>
                    <input>
                        <port id="10">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                        <port id="11">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="12">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="sum_relu" type="ReLU" precision="FP16" id="7">
                    <input>
                        <port id="13">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="14">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                </layer>
                <layer name="last" type="Convolution" precision="FP16" id="8">
                    <convolution_data
                        stride-x="1" stride-y="1"
                        pad-x="0" pad-y="0"
                        kernel-x="1" kernel-y="1"
                        output="64"
                        group="1"/>
                    <input>
                        <port id="15">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </input>
                    <output>
                        <port id="16">
                            <dim>1</dim>
                            <dim>64</dim>
                            <dim>56</dim>
                            <dim>56</dim>
                        </port>
                    </output>
                    <weights offset="156032" size="8192"/>
                    <biases offset="164224" size="128"/>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="4"/>
                <edge from-layer="3" from-port="5" to-layer="4" to-port="6"/>
                <edge from-layer="4" from-port="7" to-layer="5" to-port="8"/>
                <edge from-layer="2" from-port="3" to-layer="6" to-port="10"/>
                <edge from-layer="5" from-port="9" to-layer="6" to-port="11"/>
                <edge from-layer="6" from-port="12" to-layer="7" to-port="13"/>
                <edge from-layer="7" from-port="14" to-layer="8" to-port="15"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(164352 / sizeof(ie_fp16)));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.c_str(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weights));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo = _outputsInfo["last"];
    outputInfo->setPrecision(Precision::FP16);

    Blob::Ptr input = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, inputInfo->getDims());
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput;
    {
        SCOPED_TRACE("SW");
        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, false, input, swOutput, "input", "last"));
    }

    Blob::Ptr hwOutput;
    {
        SCOPED_TRACE("HW");
        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, true, input, hwOutput, "input", "last"));
        ASSERT_NO_FATAL_FAILURE(CheckHWRun());
    }

    Compare(hwOutput, swOutput, 18.f);
}

//
// With Permute+Flatten+Concat
//

TEST_F(myriadXHWLayersTests_nightly, PermuteFlattenConcat) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="WithPermuteFlattenConcat" version="2" batch="1">
            <layers>
                <layer name="input" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv1" type="Convolution" precision="FP16" id="2">
                    <convolution_data
                         stride-x="1"
                         stride-y="1"
                         pad-x="1"
                         pad-y="1"
                         kernel-x="3"
                         kernel-y="3"
                         output="54"
                         group="1" />
                    <input>
                        <port id="2">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>54</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </output>
                    <weights offset="0" size="248832"/>
                    <biases offset="248832" size="108"/>
                </layer>
                <layer name="perm1" type="Permute" precision="FP16" id="3">
                    <data order="0,2,3,1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>54</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </input>
                    <output>
                        <port id="5">
                            <dim>1</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                            <dim>54</dim>
                        </port>
                    </output>
                </layer>
                <layer name="flat1" type="Flatten" precision="FP16" id="4">
                    <data axis="1" end_axis="-1"/>
                    <input>
                        <port id="6">
                            <dim>1</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                            <dim>54</dim>
                        </port>
                    </input>
                    <output>
                        <port id="7">
                            <dim>1</dim>
                            <dim>28566</dim>
                        </port>
                    </output>
                </layer>

                <layer name="conv2" type="Convolution" precision="FP16" id="5">
                    <convolution_data
                         stride-x="1"
                         stride-y="1"
                         pad-x="1"
                         pad-y="1"
                         kernel-x="3"
                         kernel-y="3"
                         output="54"
                         group="1" />
                    <input>
                        <port id="8">
                            <dim>1</dim>
                            <dim>256</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </input>
                    <output>
                        <port id="9">
                            <dim>1</dim>
                            <dim>54</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </output>
                    <weights offset="0" size="248832"/>
                    <biases offset="248832" size="108"/>
                </layer>
                <layer name="perm2" type="Permute" precision="FP16" id="6">
                    <data order="0,2,3,1"/>
                    <input>
                        <port id="10">
                            <dim>1</dim>
                            <dim>54</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                        </port>
                    </input>
                    <output>
                        <port id="11">
                            <dim>1</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                            <dim>54</dim>
                        </port>
                    </output>
                </layer>
                <layer name="flat2" type="Flatten" precision="FP16" id="7">
                    <data axis="1" end_axis="-1"/>
                    <input>
                        <port id="12">
                            <dim>1</dim>
                            <dim>23</dim>
                            <dim>23</dim>
                            <dim>54</dim>
                        </port>
                    </input>
                    <output>
                        <port id="13">
                            <dim>1</dim>
                            <dim>28566</dim>
                        </port>
                    </output>
                </layer>

                <layer name="result" type="Concat" precision="FP16" id="8">
                    <concat_data axis="1"/>
                    <input>
                        <port id="14">
                            <dim>1</dim>
                            <dim>28566</dim>
                        </port>
                        <port id="15">
                            <dim>1</dim>
                            <dim>28566</dim>
                        </port>
                    </input>
                    <output>
                        <port id="16">
                            <dim>1</dim>
                            <dim>57132</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="2"/>
                <edge from-layer="2" from-port="3" to-layer="3" to-port="4"/>
                <edge from-layer="3" from-port="5" to-layer="4" to-port="6"/>
                <edge from-layer="4" from-port="7" to-layer="8" to-port="14"/>
                <edge from-layer="1" from-port="1" to-layer="5" to-port="8"/>
                <edge from-layer="5" from-port="9" to-layer="6" to-port="10"/>
                <edge from-layer="6" from-port="11" to-layer="7" to-port="12"/>
                <edge from-layer="7" from-port="13" to-layer="8" to-port="15"/>
            </edges>
        </Net>
    )V0G0N";

    TBlob<uint8_t>::Ptr weights(GenWeights(248940 / sizeof(ie_fp16)));

    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.c_str(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());
    ASSERT_NO_THROW(_net_reader.SetWeights(weights));

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    auto inputInfo = _inputsInfo["input"];
    inputInfo->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo = _outputsInfo["result"];
    outputInfo->setPrecision(Precision::FP16);

    Blob::Ptr input = make_shared_blob<ie_fp16, const SizeVector>(Precision::FP16, Layout::NCHW, inputInfo->getDims());
    input->allocate();
    GenRandomData(input);

    Blob::Ptr swOutput;
    {
        SCOPED_TRACE("SW");
        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, false, input, swOutput, "input", "result"));
    }

    Blob::Ptr hwOutput;
    {
        SCOPED_TRACE("HW");
        ASSERT_NO_FATAL_FAILURE(RunNetwork(network, true, input, hwOutput, "input", "result"));
        ASSERT_NO_FATAL_FAILURE(CheckHWRun());
    }

    Compare(hwOutput, swOutput, 1.3f);
}


TEST_F(myriadXHWLayersTests_nightly, LayerInputAligment) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    const std::string model = R"V0G0N(
        <Net name="LayerInputAligment" version="2" batch="1">
            <layers>
                <layer name="input1" type="Input" precision="FP16" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>22</dim>
                            <dim>22</dim>
                        </port>
                    </output>
                </layer>
                <layer name="input2" type="Input" precision="FP16" id="2">
                    <output>
                        <port id="2">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>24</dim>
                            <dim>24</dim>
                        </port>
                    </output>
                </layer>

                <layer name="pool1" type="Pooling" precision="FP16" id="3" >
                    <data exclude-pad="false" kernel-x="2" kernel-y="2" pad-b="0" pad-r="0" pad-x="0" pad-y="0" pool-method="max" rounding-type="ceil" stride="1,1,2,2" stride-x="2" stride-y="2"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>22</dim>
                            <dim>22</dim>
                        </port>
                    </input>
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>11</dim>
                            <dim>11</dim>
                        </port>
                    </output>
                </layer>
                <layer name="pool2" type="Pooling" precision="FP16" id="4" >
                    <data exclude-pad="false" kernel-x="2" kernel-y="2" pad-b="0" pad-r="0" pad-x="0" pad-y="0" pool-method="max" rounding-type="ceil" stride="1,1,2,2" stride-x="2" stride-y="2"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>24</dim>
                            <dim>24</dim>
                        </port>
                    </input>
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>12</dim>
                            <dim>12</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="3" to-port="0"/>
                <edge from-layer="2" from-port="2" to-layer="4" to-port="0"/>
            </edges>
        </Net>
    )V0G0N";


    ASSERT_NO_THROW(_net_reader.ReadNetwork(model.c_str(), model.length()));
    ASSERT_TRUE(_net_reader.isParseSuccess());

    auto network = _net_reader.getNetwork();

    _inputsInfo = network.getInputsInfo();
    auto inputInfo1 = _inputsInfo["input1"];
    inputInfo1->setInputPrecision(Precision::FP16);
    auto inputInfo2 = _inputsInfo["input2"];
    inputInfo2->setInputPrecision(Precision::FP16);

    _outputsInfo = network.getOutputsInfo();
    auto outputInfo1 = _outputsInfo["pool1"];
    outputInfo1->setPrecision(Precision::FP16);
    auto outputInfo2 = _outputsInfo["pool2"];
    outputInfo2->setPrecision(Precision::FP16);

    _inferRequest.reset();
    _exeNetwork.reset();

    StatusCode st;
    ASSERT_NO_THROW(st = myriadPluginPtr->LoadNetwork(_exeNetwork, network,
                                                            {{
                                                              VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION),
                                                              CONFIG_VALUE(YES)
                                                            }},
                                                          &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _exeNetwork->CreateInferRequest(_inferRequest, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = _inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
}

//
// HCW layout usage
//

TEST_F(myriadXHWLayersTests_nightly, VGG_FirstTwoConvs_HCW) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    IN_OUT_desc in_tensor, out_tensor;
    in_tensor.push_back({1, 3, 224, 224});
    out_tensor.push_back({1, 64, 224, 224});

    ParamsStruct conv1_params = {
        {"kernel-x", "3"},
        {"kernel-y", "3"},
        {"stride-x", "1"},
        {"stride-y", "1"},
        {"pad-x", "1"},
        {"pad-y", "1"},
        {"output", "64"},
        {"group", "1"}
    };
    AddLayer("Convolution",
             &conv1_params,
             1728,
             64,
             defaultWeightsRange,
             in_tensor,
             out_tensor,
             ref_convolution_wrap);

    AddLayer("ReLU",
             nullptr,
             out_tensor,
             out_tensor,
             ref_ReLU_wrap);

    ParamsStruct conv2_params = {
        {"kernel-x", "3"},
        {"kernel-y", "3"},
        {"stride-x", "1"},
        {"stride-y", "1"},
        {"pad-x", "1"},
        {"pad-y", "1"},
        {"output", "64"},
        {"group", "1"}
    };
    AddLayer("Convolution",
             &conv2_params,
             36864,
             64,
             defaultWeightsRange,
             out_tensor,
             out_tensor,
             ref_convolution_wrap);

    AddLayer("ReLU",
             nullptr,
             out_tensor,
             out_tensor,
             ref_ReLU_wrap);

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    Compare(hwOutput, swOutput, 0.7f);
}

//
// ConvPoolMerge
//

using HWConvPoolMerge = std::tuple<DimsInput,
                                   kernel, stride, pad, out_channels,
                                   kernel, stride, pad>;

class myriadXHWConvPoolMergeTests_nightly
        : public myriadXHWLayersTests_nightly,
          public testing::WithParamInterface<HWConvPoolMerge> {
public:
    tensor_test_params in_dims;
    param_size conv_kernel;
    param_size conv_stride;
    param_size conv_pad;
    size_t conv_out_c;
    param_size pool_kernel;
    param_size pool_stride;
    param_size pool_pad;

    tensor_test_params conv_out_dims;

    size_t conv_num_weights;
    size_t conv_num_biases;

    tensor_test_params pool_out_dims;

    IN_OUT_desc in_tensor, conv_out_tensor, pool_out_tensor;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(myriadXHWLayersTests_nightly::SetUp());

        in_dims = std::get<0>(GetParam());
        conv_kernel = std::get<1>(GetParam());
        conv_stride = std::get<2>(GetParam());
        conv_pad = std::get<3>(GetParam());
        conv_out_c = std::get<4>(GetParam());
        pool_kernel = std::get<5>(GetParam());
        pool_stride = std::get<6>(GetParam());
        pool_pad = std::get<7>(GetParam());

        size_t conv_out_w = (in_dims.w + 2 * conv_pad.x - conv_kernel.x + conv_stride.x) / conv_stride.x;
        size_t conv_out_h = (in_dims.h + 2 * conv_pad.y - conv_kernel.y + conv_stride.y) / conv_stride.y;
        conv_out_dims = {1, conv_out_c, conv_out_h, conv_out_w};

        conv_num_weights = conv_kernel.x * conv_kernel.y * in_dims.c * conv_out_dims.c;
        conv_num_biases = conv_out_dims.c;

        size_t pool_out_w = std::ceil((conv_out_dims.w + 2.0 * pool_pad.x - pool_kernel.x) / pool_stride.x + 1);
        size_t pool_out_h = std::ceil((conv_out_dims.h + 2.0 * pool_pad.y - pool_kernel.y) / pool_stride.y + 1);
        pool_out_dims = {1, conv_out_dims.c, pool_out_h, pool_out_w};

        in_tensor.push_back({in_dims.n, in_dims.c, in_dims.h, in_dims.w});
        conv_out_tensor.push_back({conv_out_dims.n, conv_out_dims.c, conv_out_dims.h, conv_out_dims.w});
        pool_out_tensor.push_back({pool_out_dims.n, pool_out_dims.c, pool_out_dims.h, pool_out_dims.w});
    }

    void AddConvLayer() {
        ParamsStruct conv_params = {
                  {"kernel-x", std::to_string(conv_kernel.x)}
                , {"kernel-y", std::to_string(conv_kernel.y)}
                , {"stride-x", std::to_string(conv_stride.x)}
                , {"stride-y", std::to_string(conv_stride.y)}
                , {"pad-x", std::to_string(conv_pad.x)}
                , {"pad-y", std::to_string(conv_pad.y)}
                , {"output", std::to_string(conv_out_dims.c)}
                , {"group", "1"}
        };
        AddLayer("Convolution",
                 &conv_params,
                 conv_num_weights,
                 conv_num_biases,
                 defaultWeightsRange,
                 in_tensor,
                 conv_out_tensor,
                 ref_convolution_wrap);
    }

    void AddReLULayer(float negativeSlope) {
        ParamsStruct relu_params = {
            {"negative_slope", std::to_string(negativeSlope)}
        };
        AddLayer("ReLU",
                 &relu_params,
                 conv_out_tensor,
                 conv_out_tensor,
                 ref_ReLU_wrap);
    }

    void AddPoolLayer() {
        ParamsStruct pool_params = {
                  {"kernel-x", std::to_string(pool_kernel.x)}
                , {"kernel-y", std::to_string(pool_kernel.y)}
                , {"stride-x", std::to_string(pool_stride.x)}
                , {"stride-y", std::to_string(pool_stride.y)}
                , {"pad-x", std::to_string(pool_pad.x)}
                , {"pad-y", std::to_string(pool_pad.y)}
                , {"pool-method", "max"}
        };
        AddLayer("Pooling",
                 &pool_params,
                 conv_out_tensor,
                 pool_out_tensor,
                 ref_pooling_wrap);
    }
};

TEST_P(myriadXHWConvPoolMergeTests_nightly, WithReLU) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddConvLayer();
    AddReLULayer(0.0f);
    AddPoolLayer();

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    auto maxerr = 0.0009 * in_dims.c * conv_kernel.x * conv_kernel.y;
    Compare(hwOutput, swOutput, maxerr);
}

TEST_P(myriadXHWConvPoolMergeTests_nightly, WithLeakyReLU) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddConvLayer();
    AddReLULayer(0.1f);
    AddPoolLayer();

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    auto maxerr = 0.01 * in_dims.c * conv_kernel.x * conv_kernel.y;
    Compare(hwOutput, swOutput, maxerr);
}

INSTANTIATE_TEST_CASE_P(yolo_conv1, myriadXHWConvPoolMergeTests_nightly,
                        ::testing::Combine(
                            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 448, 448)),
                            ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)),
                            ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)),
                            ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)),
                            ::testing::Values<out_channels>(16),
                            ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)),
                            ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)),
                            ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(yolo_conv2, myriadXHWConvPoolMergeTests_nightly,
                        ::testing::Combine(
                            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 224, 224)),
                            ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)),
                            ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)),
                            ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)),
                            ::testing::Values<out_channels>(32),
                            ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)),
                            ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)),
                            ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(yolo_conv4, myriadXHWConvPoolMergeTests_nightly,
                        ::testing::Combine(
                            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 56, 56)),
                            ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)),
                            ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)),
                            ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)),
                            ::testing::Values<out_channels>(128),
                            ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)),
                            ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)),
                            ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(ssd_case1, myriadXHWConvPoolMergeTests_nightly,
                        ::testing::Combine(
                            ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 98, 150)),
                            ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3)),
                            ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1)),
                            ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1)),
                            ::testing::Values<out_channels>(64),
                            ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2)),
                            ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2)),
                            ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

//
// Convolution
//

using HWConvParams = std::tuple<DimsInput, kernel, stride, pad, out_channels, group>;

class myriadXHWConvolutionTests_nightly
        : public myriadXHWLayersTests_nightly,
          public testing::WithParamInterface<HWConvParams> {
public:
    tensor_test_params in_dims;
    param_size kernel;
    param_size stride;
    param_size pad;
    size_t out_c;
    size_t group;

    tensor_test_params out_dims;

    IN_OUT_desc in_tensor, out_tensor;

    size_t numWeights;
    size_t numBiases;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(myriadXHWLayersTests_nightly::SetUp());

        in_dims = std::get<0>(GetParam());
        kernel = std::get<1>(GetParam());
        stride = std::get<2>(GetParam());
        pad = std::get<3>(GetParam());
        out_c = std::get<4>(GetParam());
        group = std::get<5>(GetParam());

        size_t out_w = (in_dims.w + 2 * pad.x - kernel.x + stride.x) / stride.x;
        size_t out_h = (in_dims.h + 2 * pad.y - kernel.y + stride.y) / stride.y;
        out_dims = {1, out_c, out_h, out_w};

        in_tensor.push_back({in_dims.n, in_dims.c, in_dims.h, in_dims.w});
        out_tensor.push_back({out_dims.n, out_dims.c, out_dims.h, out_dims.w});

        numWeights = kernel.x * kernel.y * (in_dims.c / group) * out_dims.c;
        numBiases = out_dims.c;
    }

    void AddInitialCopyLayer() {
        AddLayer("Copy",
                 nullptr,
                 in_tensor,
                 in_tensor,
                 ref_copy_wrap);
    }

    void AddConvolutionLayer() {
        std::map<std::string, std::string> convParams = {
                  {"kernel-x", std::to_string(kernel.x)}
                , {"kernel-y", std::to_string(kernel.y)}
                , {"stride-x", std::to_string(stride.x)}
                , {"stride-y", std::to_string(stride.y)}
                , {"pad-x", std::to_string(pad.x)}
                , {"pad-y", std::to_string(pad.y)}
                , {"output", std::to_string(out_c)}
                , {"group", std::to_string(group)}
        };
        AddLayer("Convolution",
                 &convParams,
                 numWeights,
                 numBiases,
                 defaultWeightsRange,
                 in_tensor,
                 out_tensor,
                 ref_convolution_wrap);
    }

    void AddReLULayer(float negativeSlope = 0.0) {
        ParamsStruct reluParams = {
            {"negative_slope", std::to_string(negativeSlope)}
        };
        AddLayer("ReLU",
                 &reluParams,
                 out_tensor,
                 out_tensor,
                 ref_ReLU_wrap);
    }

    void AddClampLayer(float min = 0.0, float max = 6.0) {
        ParamsStruct clampParams = {
                {"max", std::to_string(max)}
              , {"min", std::to_string(min)}
        };
        AddLayer("Clamp",
                 &clampParams,
                 out_tensor,
                 out_tensor,
                 ref_Clamp_wrap);
    }
};

TEST_P(myriadXHWConvolutionTests_nightly, Single) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddInitialCopyLayer();
    AddConvolutionLayer();

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    float maxerr = 0.002 * (in_dims.c / group) * kernel.x * kernel.y;
    Compare(hwOutput, swOutput, maxerr);
}

TEST_P(myriadXHWConvolutionTests_nightly, WithReLU) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddInitialCopyLayer();
    AddConvolutionLayer();
    AddReLULayer(0.0f);

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    float maxerr = 0.002 * (in_dims.c / group) * kernel.x * kernel.y;
    Compare(hwOutput, swOutput, maxerr);
}

TEST_P(myriadXHWConvolutionTests_nightly, WithLeakyReLU) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddConvolutionLayer();
    AddReLULayer(0.1f);

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    float maxerr = 0.1 * (in_dims.c / group) * kernel.x * kernel.y;
    Compare(hwOutput, swOutput, maxerr);
}

TEST_P(myriadXHWConvolutionTests_nightly, WithClamp) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddConvolutionLayer();
    AddClampLayer();

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    float maxerr = 0.1 * (in_dims.c / group) * kernel.x * kernel.y;
    Compare(hwOutput, swOutput, maxerr);
}

TEST_P(myriadXHWConvolutionTests_nightly, MultipleInfer) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddConvolutionLayer();

    SetSeed(DEFAULT_SEED_VALUE);

    _net_reader = CNNNetReader();
    _exeNetwork.reset();
    _inferRequest.reset();
    ASSERT_TRUE(GenerateNetAndInfer(true, false));

    auto outBlob = _outputMap.begin()->second;

    auto firstOutput = make_shared_blob<ie_fp16, const SizeVector>(outBlob->precision(), outBlob->layout(), outBlob->dims());
    firstOutput->allocate();
    std::copy_n(outBlob->cbuffer().as<const ie_fp16*>(), outBlob->size(), firstOutput->buffer().as<ie_fp16*>());

    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(Infer());
        ASSERT_NO_FATAL_FAILURE(Compare(outBlob, firstOutput, 0.0f)) << i;
    }
}

INSTANTIATE_TEST_CASE_P(conv_1x1s1p0_extra1, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 180, 320))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x1s1p0_extra2, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 45, 80))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(18)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x1s1p0_extra3, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 45, 80))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(294)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x1s1p0_extra4, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 45, 80))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(392)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x1s2p0_extra1, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 180, 320))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x1s2p0_extra2, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 90, 160))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x1s2p0_extra3, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 90, 160))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s2p1_extra1, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 180, 320))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s2p1_extra2, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 90, 160))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s2p1_extra3, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 45, 80))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(512)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s2p1_extra4, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 180, 320))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x1s1p0, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 56, 56))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x1s1p0_resnet50, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 2048, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(512)
                                , ::testing::Values<group>(1)
                        )
);

// This case adds extra CopyMakeBorder stage
INSTANTIATE_TEST_CASE_P(conv_1x1s1p1, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 13, 13))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(1000)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x1s2p0, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 56, 56),
                                                             MAKE_STRUCT(tensor_test_params, 1, 512, 28, 28),
                                                             MAKE_STRUCT(tensor_test_params, 1, 1024, 14, 14))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128, 256, 512, 1024)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s1p1, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 56, 56))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(192)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s1p1_yolo_tiny_v1_conv1, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 448, 448))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(16)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s1p1_yolo_tiny_v1_conv7, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(1024)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s1p1_yolo_tiny_v1_conv8, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 1024, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s1p1_vgg, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 224, 224))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_7x7s2p3, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 224, 224))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                        )
);

//  This case for unsymmetric convolution

INSTANTIATE_TEST_CASE_P(conv_3x1s1_LPR, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 22, 92))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 0))
                                , ::testing::Values<out_channels>(32)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x3s1_LPR, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 22, 92))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 1))
                                , ::testing::Values<out_channels>(32)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x5s1_LPR, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 5, 88))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 5))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_13x1s1_LPR, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 1, 88))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 13, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 6, 0))
                                , ::testing::Values<out_channels>(71)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_5x1s1_LPR, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 1, 28))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_4x4s2p1, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 256, 416))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 4, 4))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(16)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_5x5s2p1, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 256, 416))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(16)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_5x5s2p2, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 3, 256, 416))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<out_channels>(16)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s1p1_group1, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 150, 150))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(32)
                                , ::testing::Values<group>(32)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s2p1_group1, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 150, 150))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(64)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s2p1_group2, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 75, 75))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(128)
                                , ::testing::Values<group>(128)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s2p1_group3, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 38, 38))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(256)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s1p1_pva_pvd, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 6, 208, 368))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(16)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_1x1s2p0_pva_pvd, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 32, 128, 208))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(48)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(conv_3x3s1p1_ssd, myriadXHWConvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 128, 75, 75))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                        )
);

//
// Pooling
//

using HWPoolingParams = std::tuple<DimsInput, kernel, stride, pad>;

class myriadXHWPoolingTests_nightly
        : public myriadXHWLayersTests_nightly,
          public testing::WithParamInterface<HWPoolingParams> {
public:
    tensor_test_params in_dims;
    param_size kernel;
    param_size stride;
    param_size pad;

    tensor_test_params out_dims;

    IN_OUT_desc in_tensor, out_tensor;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(myriadXHWLayersTests_nightly::SetUp());

        in_dims = std::get<0>(GetParam());
        kernel = std::get<1>(GetParam());
        stride = std::get<2>(GetParam());
        pad = std::get<3>(GetParam());

        size_t out_w = std::ceil((in_dims.w + 2.0 * pad.x - kernel.x) / stride.x + 1);
        size_t out_h = std::ceil((in_dims.h + 2.0 * pad.y - kernel.y) / stride.y + 1);

        out_dims = {in_dims.n, in_dims.c, out_h, out_w};

        in_tensor.push_back({in_dims.n, in_dims.c, in_dims.h, in_dims.w});
        out_tensor.push_back({out_dims.n, out_dims.c, out_dims.h, out_dims.w});

        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    }

    void AddPoolingLayer(const std::string& poolMethod) {
        std::map<std::string, std::string> poolParams = {
                  {"kernel-x", std::to_string(kernel.x)}
                , {"kernel-y", std::to_string(kernel.y)}
                , {"stride-x", std::to_string(stride.x)}
                , {"stride-y", std::to_string(stride.y)}
                , {"pad-x", std::to_string(pad.x)}
                , {"pad-y", std::to_string(pad.y)}
                , {"pool-method", poolMethod}
        };
        AddLayer("Pooling",
                 &poolParams,
                 in_tensor,
                 out_tensor,
                 ref_pooling_wrap);
    }

    void AddReLULayer(float negativeSlope = 0.0) {
        ParamsStruct reluParams = {
            {"negative_slope", std::to_string(negativeSlope)}
        };
        AddLayer("ReLU",
                 &reluParams,
                 out_tensor,
                 out_tensor,
                 ref_ReLU_wrap);
    }

    void RunSingleTest(const std::string& poolMethod, float tolerance) {
        if (!CheckMyriadX()) {
            SKIP() << "Non-MyriadX device";
        }

        AddPoolingLayer(poolMethod);

        Blob::Ptr swOutput, hwOutput;
        ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

        Compare(hwOutput, swOutput, tolerance);
    }

    void RunWithReLUTest(const std::string& poolMethod, float tolerance) {
        if (!CheckMyriadX()) {
            SKIP() << "Non-MyriadX device";
        }

        AddPoolingLayer(poolMethod);
        AddReLULayer(0.0f);

        Blob::Ptr swOutput, hwOutput;
        ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

        Compare(hwOutput, swOutput, tolerance);
    }

    void RunMultipleInferTest(const std::string& poolMethod) {
        if (!CheckMyriadX()) {
            SKIP() << "Non-MyriadX device";
        }

        AddPoolingLayer(poolMethod);

        SetSeed(DEFAULT_SEED_VALUE);

        _net_reader = CNNNetReader();
        _exeNetwork.reset();
        _inferRequest.reset();
        ASSERT_TRUE(GenerateNetAndInfer(true, false));

        auto outBlob = _outputMap.begin()->second;

        auto firstOutput = make_shared_blob<ie_fp16, const SizeVector>(outBlob->precision(), outBlob->layout(), outBlob->dims());
        firstOutput->allocate();
        std::copy_n(outBlob->cbuffer().as<const ie_fp16*>(), outBlob->size(), firstOutput->buffer().as<ie_fp16*>());

        for (int i = 0; i < 100; ++i) {
            ASSERT_TRUE(Infer());
            ASSERT_NO_FATAL_FAILURE(Compare(outBlob, firstOutput, 0.0f)) << i;
        }
    }
};

TEST_P(myriadXHWPoolingTests_nightly, Max_Single) {
    RunSingleTest("max", 0.0f);
}

TEST_P(myriadXHWPoolingTests_nightly, Avg_Single) {
    // this case is not supported by HW
    if (kernel.x == 3 && kernel.y == 3 &&
        stride.x == 2 && stride.y == 2) {
        SKIP() << "Unsupported case";
    }
    if ((kernel.x % 2 == 0 || kernel.y % 2 == 0) &&
        (in_dims.w % 2 == 1 || in_dims.h % 2 == 1)) {
        SKIP() << "Unsupported case";
    }

    RunSingleTest("avg", 0.0013f);
}

TEST_P(myriadXHWPoolingTests_nightly, Max_WithReLU) {
    RunWithReLUTest("max", 0.0f);
}

TEST_P(myriadXHWPoolingTests_nightly, Avg_WithReLU) {
    // this case is not supported by HW
    if (kernel.x == 3 && kernel.y == 3 &&
        stride.x == 2 && stride.y == 2) {
        SKIP() << "Unsupported case";
    }
    if ((kernel.x % 2 == 0 || kernel.y % 2 == 0) &&
        (in_dims.w % 2 == 1 || in_dims.h % 2 == 1)) {
        SKIP() << "Unsupported case";
    }

    RunWithReLUTest("avg", 0.0013f);
}

TEST_P(myriadXHWPoolingTests_nightly, Max_MultipleInfer) {
    RunMultipleInferTest("max");
}

TEST_P(myriadXHWPoolingTests_nightly, Avg_MultipleInfer) {
    // this case is not supported by HW
    if (kernel.x == 3 && kernel.y == 3 &&
        stride.x == 2 && stride.y == 2) {
        SKIP() << "Unsupported case";
    }
    if ((kernel.x % 2 == 0 || kernel.y % 2 == 0) &&
        (in_dims.w % 2 == 1 || in_dims.h % 2 == 1)) {
        SKIP() << "Unsupported case";
    }

    RunMultipleInferTest("avg");
}

INSTANTIATE_TEST_CASE_P(pool_2x2s1p0, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 112, 112))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_2x2s2p0, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 224, 224),
                                                             MAKE_STRUCT(tensor_test_params, 1, 128, 112, 112),
                                                             MAKE_STRUCT(tensor_test_params, 1, 256, 56, 56),
                                                             MAKE_STRUCT(tensor_test_params, 1, 512, 28, 28),
                                                             MAKE_STRUCT(tensor_test_params, 1, 512, 14, 14))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_2x2s2p0_yolo_tiny_v1, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 16, 448, 448))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_3x3s1p0, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 192, 28, 28),
                                                             MAKE_STRUCT(tensor_test_params, 1, 100, 28, 28))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_3x3s1p1, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 192, 28, 28))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

// TODO : 3x3s2p0 HW seems to work only for Max Pooling
INSTANTIATE_TEST_CASE_P(pool_3x3s2p0, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 64, 112, 112))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_3x3s2p1, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 576, 7, 7),
                                                             MAKE_STRUCT(tensor_test_params, 1, 16, 35, 35),
                                                             MAKE_STRUCT(tensor_test_params, 1, 16, 75, 75),
                                                             MAKE_STRUCT(tensor_test_params, 1, 16, 35, 2045))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_7x7s1p0, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 1024, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_14x14s1p0, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 1024, 14, 14),
                                                             MAKE_STRUCT(tensor_test_params, 1, 1000, 14, 14))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 14, 14))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_15x15s1p0, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 1024, 15, 15),
                                                             MAKE_STRUCT(tensor_test_params, 1, 1000, 15, 15))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 15, 15))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_2x2s1p1_odd, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 512, 13, 13))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 1, 1))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_2x2s2p0_odd, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 75, 75),
                                                             MAKE_STRUCT(tensor_test_params, 2, 64, 75, 75),
                                                             MAKE_STRUCT(tensor_test_params, 2, 64, 76, 75),
                                                             MAKE_STRUCT(tensor_test_params, 2, 64, 75, 76))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_3x3s1p0_odd, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 192, 37, 37),
                                                             MAKE_STRUCT(tensor_test_params, 1, 832, 9, 9),
                                                             MAKE_STRUCT(tensor_test_params, 1, 512, 19, 19))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_3x3s2p0_odd, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 96, 93, 93),
                                                             MAKE_STRUCT(tensor_test_params, 1, 512, 23, 23),
                                                             MAKE_STRUCT(tensor_test_params, 1, 192, 75, 75),
                                                             MAKE_STRUCT(tensor_test_params, 1, 480, 37, 37))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_3x3s2p0_extra, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 96, 32, 52))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

INSTANTIATE_TEST_CASE_P(pool_7x7s7p0_rfcn_batch, myriadXHWPoolingTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 300, 5, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 7, 7))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                        )
);

//
// TF
//

PRETTY_PARAM(tfPad, paddings4)
using ConvTFParams = std::tuple<DimsInput, DimsOutput, kernel, stride, tfPad, group>;

class myriadXHWConvTFTests_nightly :
        public myriadXHWLayersTests_nightly,
        public testing::WithParamInterface<ConvTFParams>{
public:
    tensor_test_params inDims;
    tensor_test_params outDims;
    param_size kernel;
    param_size stride;
    paddings4 pad;
    int group;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(myriadXHWLayersTests_nightly::SetUp());

        inDims = std::get<0>(GetParam());
        outDims = std::get<1>(GetParam());
        kernel = std::get<2>(GetParam());
        stride = std::get<3>(GetParam());
        pad = std::get<4>(GetParam());
        group = std::get<5>(GetParam());
    }

    void AddConvolutionLayer() {
        std::map<std::string, std::string> convParams = {
            {"kernel-x", std::to_string(kernel.x)},
            {"kernel-y", std::to_string(kernel.y)},
            {"stride-x", std::to_string(stride.x)},
            {"stride-y", std::to_string(stride.y)},
            {"auto_pad", "same_upper"},
            {"pad-x", std::to_string(pad.left)},
            {"pad-r", std::to_string(pad.right)},
            {"pad-y", std::to_string(pad.top)},
            {"pad-b", std::to_string(pad.bottom)},
            {"dilation-x", "1"},
            {"dilation-y", "1"},
            {"group", std::to_string(group)},
            {"output", std::to_string(outDims.c)}
        };

        AddLayer(
            "Convolution",
            &convParams,
            kernel.x * kernel.y * (inDims.c / group) * outDims.c,
            outDims.c,
            defaultWeightsRange,
            {{inDims.n, inDims.c, inDims.h, inDims.w}},
            {{outDims.n, outDims.c, outDims.h, outDims.w}},
            ref_convolution_wrap);
    }
};

TEST_P(myriadXHWConvTFTests_nightly, Single) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddConvolutionLayer();

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    float maxerr = 0.002 * (inDims.c / group) * kernel.x * kernel.y;
    Compare(hwOutput, swOutput, maxerr);
}

INSTANTIATE_TEST_CASE_P(tf, myriadXHWConvTFTests_nightly,
    ::testing::Values(
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 3, 224, 224),    // input
            MAKE_STRUCT(tensor_test_params, 1, 24, 112, 112),   // output
            MAKE_STRUCT(param_size, 7, 7),                      // kernel
            MAKE_STRUCT(param_size, 2, 2),                      // stride
            MAKE_STRUCT(paddings4, 2, 2, 3, 3),                 // pad
            3                                                   // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 64, 56, 56),     // input
            MAKE_STRUCT(tensor_test_params, 1, 192, 56, 56),    // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 1, 1),                      // stride
            MAKE_STRUCT(paddings4, 1, 1, 1, 1),                 // pad
            1                                                   // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 128, 3, 3),      // input
            MAKE_STRUCT(tensor_test_params, 1, 128, 2, 2),      // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 2, 2),                      // stride
            MAKE_STRUCT(paddings4, 1, 1, 1, 1),                 // pad
            128                                                 // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 256, 2, 2),      // input
            MAKE_STRUCT(tensor_test_params, 1, 24, 2, 2),       // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 1, 1),                      // stride
            MAKE_STRUCT(paddings4, 1, 1, 1, 1),                 // pad
            1                                                   // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 64, 2, 2),       // input
            MAKE_STRUCT(tensor_test_params, 1, 64, 1, 1),       // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 2, 2),                      // stride
            MAKE_STRUCT(paddings4, 0, 0, 1, 1),                 // pad
            64                                                  // group
        ),
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 128, 1, 1),      // input
            MAKE_STRUCT(tensor_test_params, 1, 24, 1, 1),       // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 1, 1),                      // stride
            MAKE_STRUCT(paddings4, 1, 1, 1, 1),                 // pad
            1                                                   // group
        )
    )
);

using PoolTFParams = std::tuple<DimsInput, DimsOutput, kernel, stride, tfPad>;

class myriadXHWPoolTFTests_nightly :
        public myriadXHWLayersTests_nightly,
        public testing::WithParamInterface<PoolTFParams>{
public:
    tensor_test_params inDims;
    tensor_test_params outDims;
    param_size kernel;
    param_size stride;
    paddings4 pad;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(myriadXHWLayersTests_nightly::SetUp());

        inDims = std::get<0>(GetParam());
        outDims = std::get<1>(GetParam());
        kernel = std::get<2>(GetParam());
        stride = std::get<3>(GetParam());
        pad = std::get<4>(GetParam());
    }

    void AddPoolingLayer() {
        std::map<std::string, std::string> poolParams = {
            {"pool-method", "max"},
            {"kernel-x", std::to_string(kernel.x)},
            {"kernel-y", std::to_string(kernel.y)},
            {"stride-x", std::to_string(stride.x)},
            {"stride-y", std::to_string(stride.y)},
            {"auto_pad", "same_upper"},
            {"exclude-pad", "true"},
            {"rounding-type", "floor"},
            {"pad-x", std::to_string(pad.left)},
            {"pad-r", std::to_string(pad.right)},
            {"pad-y", std::to_string(pad.top)},
            {"pad-b", std::to_string(pad.bottom)}
        };

        AddLayer(
            "Pooling",
            &poolParams,
            {{inDims.n, inDims.c, inDims.h, inDims.w}},
            {{outDims.n, outDims.c, outDims.h, outDims.w}},
            ref_pooling_wrap);
    }
};

TEST_P(myriadXHWPoolTFTests_nightly, Single) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddPoolingLayer();

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    Compare(hwOutput, swOutput, 0.0f);
}

INSTANTIATE_TEST_CASE_P(tf, myriadXHWPoolTFTests_nightly,
    ::testing::Values(
        std::make_tuple(
            MAKE_STRUCT(tensor_test_params, 1, 64, 112, 112),   // input
            MAKE_STRUCT(tensor_test_params, 1, 64, 56, 56),     // output
            MAKE_STRUCT(param_size, 3, 3),                      // kernel
            MAKE_STRUCT(param_size, 2, 2),                      // stride
            MAKE_STRUCT(paddings4, 0, 0, 1, 1)                  // pad
        )
    )
);

//
// FullyConnected
//

class myriadXHWFullyConnected_nightly
        : public myriadXHWLayersTests_nightly,
          public testing::WithParamInterface<fcon_test_params> {
public:
    fcon_test_params p;

    size_t numWeights;
    size_t numBias;

    IN_OUT_desc in_tensor, out_tensor;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(myriadXHWLayersTests_nightly::SetUp());

        p = GetParam();

        numWeights = p.in.c * p.in.h * p.in.w * p.out_c;
        numBias = p.out_c;

        in_tensor.push_back({p.in.n, p.in.c, p.in.h, p.in.w});
        out_tensor.push_back({1, p.out_c});
    }

    void AddFCLayer() {
        std::map<std::string, std::string> fcParams;
        fcParams["out-size"] = std::to_string(p.out_c);
        AddLayer("FullyConnected",
                 &fcParams,
                 numWeights,
                 numBias,
                 defaultWeightsRange,
                 in_tensor,
                 out_tensor,
                 ref_innerproduct_wrap);
    }

    void AddReLULayer(float negativeSlope = 0.0) {
        ParamsStruct reluParams = {
            {"negative_slope", std::to_string(negativeSlope)}
        };
        AddLayer("ReLU",
                 &reluParams,
                 out_tensor,
                 out_tensor,
                 ref_ReLU_wrap);
    }
};

TEST_P(myriadXHWFullyConnected_nightly, Single) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddFCLayer();

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    Compare(hwOutput, swOutput, p.error_bound);
}

TEST_P(myriadXHWFullyConnected_nightly, WithReLU) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddFCLayer();
    AddReLULayer(0.0f);

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    Compare(hwOutput, swOutput, p.error_bound);
}

TEST_P(myriadXHWFullyConnected_nightly, MultipleInfer) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddFCLayer();

    SetSeed(DEFAULT_SEED_VALUE);

    _net_reader = CNNNetReader();
    _exeNetwork.reset();
    _inferRequest.reset();
    ASSERT_TRUE(GenerateNetAndInfer(true, false));

    auto outBlob = _outputMap.begin()->second;

    auto firstOutput = make_shared_blob<ie_fp16, const SizeVector>(outBlob->precision(), outBlob->layout(), outBlob->dims());
    firstOutput->allocate();
    std::copy_n(outBlob->cbuffer().as<const ie_fp16*>(), outBlob->size(), firstOutput->buffer().as<ie_fp16*>());

    for (int i = 0; i < 100; ++i) {
        ASSERT_TRUE(Infer());
        ASSERT_NO_FATAL_FAILURE(Compare(outBlob, firstOutput, 0.0f)) << i;
    }
}

INSTANTIATE_TEST_CASE_P(fc_1024to1000, myriadXHWFullyConnected_nightly,
        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 1024, 1, 1}, 1000, 0.25f))
);

INSTANTIATE_TEST_CASE_P(fc_4096to1000, myriadXHWFullyConnected_nightly,
        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 4096, 1, 1}, 1000, 0.82f))
);

INSTANTIATE_TEST_CASE_P(fc_4096to4096, myriadXHWFullyConnected_nightly,
        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 4096, 1, 1}, 4096, 0.82f))
);

INSTANTIATE_TEST_CASE_P(fc_16x16x16to16, myriadXHWFullyConnected_nightly,
        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 16, 16, 16}, 16, 0.71f))
);

INSTANTIATE_TEST_CASE_P(fc_512x7x7to4096, myriadXHWFullyConnected_nightly,
        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 512, 7, 7}, 4096, 4.38f))
);

INSTANTIATE_TEST_CASE_P(fc_256x7x7to1470, myriadXHWFullyConnected_nightly,
        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 256, 7, 7}, 1470, 2.375f))
);

INSTANTIATE_TEST_CASE_P(fc_576to128, myriadXHWFullyConnected_nightly,
        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 576, 1, 1}, 128, 0.76f))
);

INSTANTIATE_TEST_CASE_P(fc_1152to128, myriadXHWFullyConnected_nightly,
        ::testing::Values(MAKE_STRUCT(fcon_test_params, {1, 1152, 1, 1}, 128, 0.76f))
);


//HW_DECONV


using HWDeconvParams = std::tuple<DimsInput, kernel, stride, pad, out_channels, group>;

class myriadXHWDeconvolutionTests_nightly
        : public myriadXHWLayersTests_nightly,
          public testing::WithParamInterface<HWDeconvParams> {
public:
    tensor_test_params in_dims;
    param_size kernel;
    param_size stride;
    param_size pad;
    size_t out_c;
    size_t group;

    tensor_test_params out_dims;

    IN_OUT_desc in_tensor, out_tensor;

    size_t numWeights;
    size_t numBiases;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(myriadXHWLayersTests_nightly::SetUp());

        in_dims = std::get<0>(GetParam());
        kernel = std::get<1>(GetParam());
        stride = std::get<2>(GetParam());
        pad = std::get<3>(GetParam());
        out_c = std::get<4>(GetParam());
        group = std::get<5>(GetParam());

        size_t out_w = stride.x * (in_dims.w - 1) + kernel.x - 2 * pad.x;
        size_t out_h = stride.y * (in_dims.h - 1) + kernel.y - 2 * pad.y;
        out_dims = {1, out_c, out_h, out_w};

        in_tensor.push_back({in_dims.n, in_dims.c, in_dims.h, in_dims.w});
        out_tensor.push_back({out_dims.n, out_dims.c, out_dims.h, out_dims.w});

        numWeights = kernel.x * kernel.y * (in_dims.c / group) * out_dims.c;
        numBiases = out_dims.c;
    }

    void AddInitialCopyLayer() {
        AddLayer("Copy",
                 nullptr,
                 in_tensor,
                 in_tensor,
                 ref_copy_wrap);
    }

    void AddDeconvolutionLayer() {
        std::map<std::string, std::string> deconvParams = {
                  {"kernel-x", std::to_string(kernel.x)}
                , {"kernel-y", std::to_string(kernel.y)}
                , {"stride-x", std::to_string(stride.x)}
                , {"stride-y", std::to_string(stride.y)}
                , {"pad-x", std::to_string(pad.x)}
                , {"pad-y", std::to_string(pad.y)}
                , {"output", std::to_string(out_c)}
                , {"group", std::to_string(group)}
        };
        AddLayer("Deconvolution",
                 &deconvParams,
                 numWeights,
                 numBiases,
                 defaultWeightsRange,
                 in_tensor,
                 out_tensor,
                 ref_deconvolution_wrap);
    }

    void AddDeconvolutionLayerSmallWeights() {
        std::map<std::string, std::string> deconvParams = {
                  {"kernel-x", std::to_string(kernel.x)}
                , {"kernel-y", std::to_string(kernel.y)}
                , {"stride-x", std::to_string(stride.x)}
                , {"stride-y", std::to_string(stride.y)}
                , {"pad-x", std::to_string(pad.x)}
                , {"pad-y", std::to_string(pad.y)}
                , {"output", std::to_string(out_c)}
                , {"group", std::to_string(group)}
        };
        AddLayer("Deconvolution",
                 &deconvParams,
                 numWeights,
                 numBiases,
                 smallWeightsRange,
                 in_tensor,
                 out_tensor,
                 ref_deconvolution_wrap);
    }
};

class myriadXHWDeconvolutionTestsScale : public myriadXHWDeconvolutionTests_nightly {
};

TEST_P(myriadXHWDeconvolutionTests_nightly, Single) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddInitialCopyLayer();
    AddDeconvolutionLayer();

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    float maxerr = 0.002 * (in_dims.c / group) * kernel.x * kernel.y;
    Compare(hwOutput, swOutput, maxerr);
}

TEST_P(myriadXHWDeconvolutionTestsScale, ScaleTests) {
    if (!CheckMyriadX()) {
        SKIP() << "Non-MyriadX device";
    }

    AddInitialCopyLayer();
    AddDeconvolutionLayerSmallWeights();

    Blob::Ptr swOutput, hwOutput;
    ASSERT_NO_FATAL_FAILURE(RunBothCases(swOutput, hwOutput));

    float maxerr = 0.01;
    Compare(hwOutput, swOutput, maxerr);
}

INSTANTIATE_TEST_CASE_P(deconv_tf_ssd, myriadXHWDeconvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 2, 3, 3))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(1)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(deconv_3x3_str1, myriadXHWDeconvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 3, 3),
                                                             MAKE_STRUCT(tensor_test_params, 1, 128, 5, 5),
                                                             MAKE_STRUCT(tensor_test_params, 1, 64, 7, 7))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0))
                                , ::testing::Values<out_channels>(128, 256)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(hw_accuracy_deconv_3x3, myriadXHWDeconvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 256, 5, 5),
                                                             MAKE_STRUCT(tensor_test_params, 1, 128, 11, 11),
                                                             MAKE_STRUCT(tensor_test_params, 1, 64, 13, 13),
                                                             MAKE_STRUCT(tensor_test_params, 1, 32, 8, 8))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 3, 3))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0),
                                                         MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(1, 128)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(hw_accuracy_deconv, myriadXHWDeconvolutionTests_nightly,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 120, 36, 36),
                                                             MAKE_STRUCT(tensor_test_params, 1, 73, 40, 54),
                                                             MAKE_STRUCT(tensor_test_params, 1, 7, 9, 13))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1),
                                                            MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0),
                                                         MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(19, 53)
                                , ::testing::Values<group>(1)
                        )
);

INSTANTIATE_TEST_CASE_P(hw_accuracy_scale_deconv, myriadXHWDeconvolutionTestsScale,
                        ::testing::Combine(
                                ::testing::Values<DimsInput>(MAKE_STRUCT(tensor_test_params, 1, 120, 36, 36))
                                , ::testing::Values<kernel>(MAKE_STRUCT(param_size, 5, 5))
                                , ::testing::Values<stride>(MAKE_STRUCT(param_size, 1, 1),
                                                            MAKE_STRUCT(param_size, 2, 2))
                                , ::testing::Values<pad>(MAKE_STRUCT(param_size, 0, 0),
                                                         MAKE_STRUCT(param_size, 1, 1))
                                , ::testing::Values<out_channels>(256)
                                , ::testing::Values<group>(1)
                        )
);
