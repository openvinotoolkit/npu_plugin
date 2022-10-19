//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_LRN.3720xx.text.xdat"

#include "param_lrn.h"

#define MAX_LRN_PARAMS 3

struct LRNTest {
    Dims inputDims;
    Dims axisDims;
    Dims outputDims;
    StorageOrder storageOrder;
    const char* kernelName;
    float params[MAX_LRN_PARAMS];
    CustomParams customLayerParams;
};

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, LRN)) {
    static constexpr std::initializer_list<LRNTest> lrn_test_list{
            {{4, 10, 4},
             {1},
             {4, 10, 4},
             orderYXZ,
             FPE("lrn.elf"),
             {0.5 /*alpha*/, 0.5 /*beta*/, 0.5 /*bias*/},
             {{
                     3 /*size*/, sw_params::Location::NN_CMX /*mem type*/, 1, /*axis*/
             }}},
            {{2, 3, 5},
             {2},
             {2, 3, 5},
             orderZYX,
             FPE("lrn.elf"),
             {0.5 /*alpha*/, 0.5 /*beta*/, 0.5 /*bias*/},
             {{
                     4 /*size*/, sw_params::Location::NN_CMX /*mem type*/, 0, 2, /*axis*/
             }}},
            {{3, 4, 5},
             {2},
             {3, 4, 5},
             orderXYZ,
             FPE("lrn.elf"),
             {0.5 /*alpha*/, 0.5 /*beta*/, 0.5 /*bias*/},
             {{
                     3 /*size*/, sw_params::Location::NN_CMX /*mem type*/, 1, 2, /*axis*/
             }}},
            {{4, 5, 6},
             {1},
             {4, 5, 6},
             orderYZX,
             FPE("lrn.elf"),
             {0.5 /*alpha*/, 0.5 /*beta*/, 0.5 /*bias*/},
             {{
                     2 /*size*/, sw_params::Location::NN_CMX /*mem type*/, 2, /*axis*/
             }}},
            {{6, 7, 8},
             {2},
             {6, 7, 8},
             orderXZY,
             FPE("lrn.elf"),
             {0.5 /*alpha*/, 0.5 /*beta*/, 0.5 /*bias*/},
             {{
                     5 /*size*/, sw_params::Location::NN_CMX /*mem type*/, 0, 2, /*axis*/
             }}},
            {{1, 2, 3},
             {2},
             {1, 2, 3},
             orderZXY,
             FPE("lrn.elf"),
             {0.5 /*alpha*/, 0.5 /*beta*/, 0.5 /*bias*/},
             {{
                     1 /*size*/, sw_params::Location::NN_CMX /*mem type*/, 0, 1, /*axis*/
             }}},
    };

    class CustomCppLRNTest : public CustomCppTests<fp16, LRNTest> {
    public:
        explicit CustomCppLRNTest(): m_testsLoop(lrn_test_list, "test") {
        }
        virtual ~CustomCppLRNTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppLRNTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, emptyParamData, MAX_LOCAL_PARAMS, 0};

            CustomCppTests<fp16, LRNTest>::initData();
            const LRNTest* test = m_currentTest;
            int32_t ind[subspace::MAX_DIMS] = {0};
            subspace::orderToIndices((t_D8StorageOrder)(test->storageOrder), ind);
            m_size = static_cast<int64_t>(test->customLayerParams.layerParams[0]);
            m_alpha = test->params[0];
            m_beta = test->params[1];
            m_bias = test->params[2];
            m_lrnParams = reinterpret_cast<sw_params::LRNParams*>(paramContainer);
            *m_lrnParams = sw_params::LRNParams();
            m_lrnParams->alpha = m_alpha;
            m_lrnParams->beta = m_beta;
            m_lrnParams->bias = m_bias;
            m_lrnParams->size = m_size;
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::LRNParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[1]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_lrnParams);

            const auto axisSize = test->axisDims.begin()[0];
            for (int i = 0; i < axisSize; ++i) {
                m_axisVector.push_back(test->customLayerParams.layerParams[i + 2]);
            }

            const Dims& axisDims = test->axisDims;
            const TensorDims dims3Axis(axisDims.begin()[0]);
            m_axisTensor.init(StorageOrder(orderW), dims3Axis);

            allocBuffer(m_axisTensor);

            m_params.kernel = reinterpret_cast<uint64_t>(sk_single_shave_LRN_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.008f;
        }

        void initParserRunner() override {
            initMyriadResources();
            initDebugInfo();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
            OpTensor inBuff;
            OpTensor axisBuff;
            OpTensor outBuff;
            m_inputTensor.exportToBuffer(inBuff);
            m_axisTensor.exportToBuffer(axisBuff);
            m_outputTensor.exportToBuffer(outBuff);

            customCppOp->addInputBuffer(inBuff, m_requiredTensorLocation);
            customCppOp->addInputBuffer(axisBuff, m_requiredTensorLocation);
            customCppOp->addOutputBuffer(outBuff, m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        void generateInputData() override {
            rand_seed();

            // set random seed
            u64 ticks_for_seed = rtems_clock_get_uptime_nanoseconds();
            srand(ticks_for_seed);

            // input
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                float tmp = float(rand() % 600) / 100 - 3.0f;
                m_inputTensor.at(indices) = f32Tof16(tmp);
            });
            int indice = 0;
            m_axisTensor.forEach(false, [&](const MemoryDims& indices) {
                m_axisTensor.at(indices) = m_axisVector[indice++];
            });
        }

        void generateReferenceData() override {
            const auto& dims = m_inputTensor.tensorDims();
            const auto axisdims = m_axisTensor.tensorDims();
            const int axisSize = axisdims.width;
            // Force the layout of the axisFlag is NCHW.
            std::vector<int32_t> axisFlag(4, 0);
            axisFlag = checkAxis(m_axisTensor, axisFlag, axisSize);
            auto batch = dims.batch == 0 ? 1 : dims.batch;
            for (int n = 0; n < batch; ++n) {
                for (int c = 0; c < dims.channels; ++c) {
                    for (int h = 0; h < dims.height; ++h) {
                        for (int w = 0; w < dims.width; ++w) {
                            m_referenceOutputTensor.at(TensorDims(w, h, c, n)) =
                                    calculate_lrn(w, h, c, n, axisFlag, axisSize, batch);
                        }
                    }
                }
            }
        }
        half calculate_lrn(int w, int h, int c, int n, std::vector<int32_t> axisFlag, int axisSize, int batch) {
            const int size = static_cast<int>(m_size);
            const float alpha = static_cast<float>(m_alpha);
            const float beta = static_cast<float>(m_beta);
            const float bias = static_cast<float>(m_bias);

            int fromN = axisFlag[0] ? std::max(n - (size - 1) / 2, 0) : n;
            int toN = axisFlag[0] ? std::min(n + (size - 1) / 2 + 1, batch) : n + 1;
            int fromW = axisFlag[3] ? std::max(w - (size - 1) / 2, 0) : w;
            int toW = axisFlag[3] ? std::min(w + (size - 1) / 2 + 1, m_inputTensor.tensorDims().width) : w + 1;
            int fromH = axisFlag[2] ? std::max(h - (size - 1) / 2, 0) : h;
            int toH = axisFlag[2] ? std::min(h + (size - 1) / 2 + 1, m_inputTensor.tensorDims().height) : h + 1;
            int fromC = axisFlag[1] ? std::max(c - (size - 1) / 2, 0) : c;
            int toC = axisFlag[1] ? std::min(c + (size - 1) / 2 + 1, m_inputTensor.tensorDims().channels) : c + 1;

            float totalSum = 0;
            for (int posN = fromN; posN < toN; ++posN) {
                for (int posW = fromW; posW < toW; ++posW) {
                    for (int posH = fromH; posH < toH; ++posH) {
                        for (int posC = fromC; posC < toC; ++posC) {
                            float val = f16Tof32(m_inputTensor.at(TensorDims(posW, posH, posC, posN)));
                            totalSum += val * val;
                        }
                    }
                }
            }
            float ref = f16Tof32(m_inputTensor.at(TensorDims(w, h, c, n))) /
                        pow(bias + alpha * totalSum / (float)(pow(size, axisSize)), beta);
            return f32Tof16(ref);
        }
        std::vector<int32_t> checkAxis(Tensor<int32_t> m_axisTensor, std::vector<int32_t> axisFlag, int axisSize) {
            auto order = m_inputTensor.storageOrder();
            int i;
            switch (order) {
            case orderHWC:
                for (i = 0; i < axisSize; ++i) {
                    if (0 == m_axisTensor.at(TensorDims(i)) ||
                        0 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[2] = 1;
                    } else if (1 == m_axisTensor.at(TensorDims(i)) ||
                               1 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[3] = 1;
                    } else if (2 == m_axisTensor.at(TensorDims(i)) ||
                               2 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[1] = 1;
                    }
                }
                break;
            case orderCHW:
                for (i = 0; i < axisSize; ++i) {
                    if (0 == m_axisTensor.at(TensorDims(i)) ||
                        0 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[1] = 1;
                    } else if (1 == m_axisTensor.at(TensorDims(i)) ||
                               1 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[2] = 1;
                    } else if (2 == m_axisTensor.at(TensorDims(i)) ||
                               2 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[3] = 1;
                    }
                }
                break;
            case orderWHC:
                for (int i = 0; i < axisSize; ++i) {
                    if (0 == m_axisTensor.at(TensorDims(i)) ||
                        0 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[3] = 1;
                    } else if (1 == m_axisTensor.at(TensorDims(i)) ||
                               1 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[2] = 1;
                    } else if (2 == m_axisTensor.at(TensorDims(i)) ||
                               2 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[1] = 1;
                    }
                }
                break;
            case orderHCW:
                for (int i = 0; i < axisSize; ++i) {
                    if (0 == m_axisTensor.at(TensorDims(i)) ||
                        0 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[2] = 1;
                    } else if (1 == m_axisTensor.at(TensorDims(i)) ||
                               1 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[1] = 1;
                    } else if (2 == m_axisTensor.at(TensorDims(i)) ||
                               2 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[3] = 1;
                    }
                }
                break;
            case orderWCH:
                for (int i = 0; i < axisSize; ++i) {
                    if (0 == m_axisTensor.at(TensorDims(i)) ||
                        0 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[3] = 1;
                    } else if (1 == m_axisTensor.at(TensorDims(i)) ||
                               1 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[1] = 1;
                    } else if (2 == m_axisTensor.at(TensorDims(i)) ||
                               2 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[2] = 1;
                    }
                }
                break;
            case orderCWH:
                for (int i = 0; i < axisSize; ++i) {
                    if (0 == m_axisTensor.at(TensorDims(i)) ||
                        0 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[1] = 1;
                    } else if (1 == m_axisTensor.at(TensorDims(i)) ||
                               1 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[3] = 1;
                    } else if (2 == m_axisTensor.at(TensorDims(i)) ||
                               2 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[2] = 1;
                    }
                }
                break;

            case orderNHWC:
                for (int i = 0; i < axisSize; ++i) {
                    if (0 == m_axisTensor.at(TensorDims(i)) ||
                        0 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[0] = 1;
                    } else if (1 == m_axisTensor.at(TensorDims(i)) ||
                               1 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[2] = 1;
                    } else if (2 == m_axisTensor.at(TensorDims(i)) ||
                               2 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[3] = 1;
                    } else if (3 == m_axisTensor.at(TensorDims(i)) ||
                               3 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[1] = 1;
                    }
                }
                break;
            case orderNHCW:
                for (int i = 0; i < axisSize; ++i) {
                    if (0 == m_axisTensor.at(TensorDims(i)) ||
                        0 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[0] = 1;
                    } else if (1 == m_axisTensor.at(TensorDims(i)) ||
                               1 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[2] = 1;
                    } else if (2 == m_axisTensor.at(TensorDims(i)) ||
                               2 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[1] = 1;
                    } else if (3 == m_axisTensor.at(TensorDims(i)) ||
                               3 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[3] = 1;
                    }
                }
                break;
            case orderNCHW:
                for (int i = 0; i < axisSize; ++i) {
                    if (0 == m_axisTensor.at(TensorDims(i)) ||
                        0 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[0] = 1;
                    } else if (1 == m_axisTensor.at(TensorDims(i)) ||
                               1 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[1] = 1;
                    } else if (2 == m_axisTensor.at(TensorDims(i)) ||
                               2 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[2] = 1;
                    } else if (3 == m_axisTensor.at(TensorDims(i)) ||
                               3 == (m_axisTensor.at(TensorDims(i)) + m_inputTensor.ndims())) {
                        axisFlag[3] = 1;
                    }
                }
                break;
            }
            return axisFlag;
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();

            // save output data
            if (m_save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                 "outMyriad.bin");
            }

            bool threshold_test_failed = false;

            m_outputTensor.forEach(false, [&](const MemoryDims& indices) {
                float input = f16Tof32(m_inputTensor.at(indices));
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= m_test_threshold);

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f %f\n", ti.height, ti.width, ti.channels, input, value,
                           gt_value, abs_diff);
                }
            });
            return !threshold_test_failed;
        }

    private:
        ListIterator<LRNTest> m_testsLoop;

        float m_alpha;
        float m_beta;
        float m_bias;
        int64_t m_size;
        std::vector<int32_t> m_axisVector;
        Tensor<int32_t> m_axisTensor;

        sw_params::LRNParams* m_lrnParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppLRNTest)
}  // namespace )
