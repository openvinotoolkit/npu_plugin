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

#include <custom_cpp_tests.h>
#include <random>
#include "layers/param_custom_cpp.h"
#include "mvSubspaces.h"

#ifdef CONFIG_TARGET_SOC_3720
__attribute__((aligned(1024)))
#include "sk.broadcast.3010xx.text.xdat"
#else
#include "svuSLKernels_EP.h"
#endif

#include "param_broadcast.h"

#define SEED_VALUE 0xbdd1cb13  // defined to use this value as random seed
typedef std::initializer_list<int32_t> targetShape;
typedef std::initializer_list<int32_t> axesMapping;

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, Broadcast)) {
    static constexpr std::initializer_list<SingleTest> broadcast_test_list {
            {{1}, // input dims
             {},   // output dims
             orderZYX,
             FPE("broadcast.elf"),
             {{
                    sw_params::BroadcastType::NUMPY /*Broadcast mode*/,
                    sw_params::Location::NN_CMX /*mem type*/,
             }}},
    };

    static constexpr std::initializer_list<targetShape> targetShape_list {
        {1000, 56, 7, 7}
    };

    static constexpr std::initializer_list<axesMapping> axesMapping_list {
        {}
    };

    // static constexpr std::initializer_list<BroadcastTestParams> dimensions_list = {
    //     { { 1                }, { 1000, 56, 7, 7        }, {}, NUMPY },
    //     { { 1                }, { 1440                  }, {}, NUMPY },
    //     { { 1                }, { 1440,  2              }, {}, NUMPY },
    //     { { 1                }, {   14, 15, 16, 2       }, {}, NUMPY },
    //     { { 14               }, {   14, 15, 16, 2       }, {}, NUMPY },
    //     { {  1, 15           }, {   14, 15, 16, 2       }, {}, NUMPY },
    //     { { 14, 15           }, {   14, 15, 16, 2       }, {}, NUMPY },
    //     { {  1,  1, 16       }, {   14, 15, 16, 2       }, {}, NUMPY },
    //     { { 14,  1, 16       }, {   14, 15, 16, 2       }, {}, NUMPY },
    //     { {  1, 15, 16       }, {   14, 15, 16, 2       }, {}, NUMPY },
    // //    { {  1               }, {   80                  }, { 0 }, EXPLICIT },
    //     { { 16               }, {   50, 50, 16, 1       }, { 1 }, EXPLICIT },
    //     { { 50, 50           }, {   16, 50, 50, 1       }, { 1, 2 }, EXPLICIT },
    // //    { {  0               }, { 1440                  }, {}, NUMPY },
    // //    { {  1               }, { 0                     }, {}, NUMPY },
    // //    { {  0               }, { 0                     }, {}, NUMPY },
    // //    { {  0, 50           }, {   16, 50, 50, 1       }, { 1, 2 }, EXPLICIT },
    // //    { { 50, 50           }, {   16, 50, 50, 0       }, { 1, 2 }, EXPLICIT },
    //     { { 1                }, { 1000, 56, 7, 7        }, {}, BIDIRECTIONAL },
    //     { { 1                }, { 1440                  }, {}, BIDIRECTIONAL },
    //     { { 14, 15           }, {   14, 15, 16, 2       }, {}, BIDIRECTIONAL },
    //     { { 14, 15, 16, 2    }, {   14,  1, 16          }, {}, BIDIRECTIONAL },
    //     { { 14,  1, 16       }, {    1, 15,  1, 2       }, {}, BIDIRECTIONAL },
    //  //   { { 0                }, { 1440                  }, {}, BIDIRECTIONAL },
    //  //   { { 1                }, { 0                     }, {}, BIDIRECTIONAL },
    // };

    class CustomCppBroadcastTest : public CustomCppTests<fp16> {
    public:
        explicit CustomCppBroadcastTest(): m_testsLoop(broadcast_test_list, "test"),
            m_targetShape(targetShape_list),
            m_axesMapping(axesMapping_list)  {}
        virtual ~CustomCppBroadcastTest() {}

    protected:
        const char* suiteName() const override {
            return "CustomCppBroadcastTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
            addLoop(m_targetShape);
            addLoop(m_axesMapping);
        }

        void initData() override {
            m_params = {0xFFFFFFFF, m_elfBuffer, 0, nullptr, MAX_LOCAL_PARAMS, 0, 0};

            initElfBuffer();
            initTestCase();
            const Dimensions& dimIn = m_currentTest->inDim;
            // TODO: Target shape and axes mapping dims?
            const Dimensions& dimTargetShape = m_currentTest->inDim;
            const Dimensions& dimAxesMapping = m_currentTest->inDim;
            const Dimensions& dimOut = m_currentTest->outDim;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;

            const TensorDims dims3In(dimIn.width,   dimIn.height,  dimIn.channels,  1);
            // TODO: dim3 targes shape and axes mapping?
            const TensorDims dims3TargetShape(dimTargetShape.width,   dimTargetShape.height,  dimTargetShape.channels,  1);
            const TensorDims dims3AxesMappins(dimAxesMapping.width,   dimAxesMapping.height,  dimAxesMapping.channels,  1);
            const TensorDims dims3Out(dimOut.width, dimOut.height, dimOut.channels, 1);

            m_inputTensor.init(storageOrder, dims3In);
            m_targetShapeTensor.init(storageOrder, dims3TargetShape);
            m_axesMappingTensor.init(storageOrder, dims3AxesMappins);
            m_outputTensor.init(storageOrder, dims3Out);
            m_referenceOutputTensor.init(storageOrder, dims3Out);

            allocBuffer(m_inputTensor);
            allocBuffer(m_targetShapeTensor);
            allocBuffer(m_axesMappingTensor);
            allocBuffer(m_outputTensor);
            allocBuffer(m_referenceOutputTensor);

           const SingleTest* test = m_currentTest;
           m_mode = static_cast<sw_params::BroadcastType>(test->customLayerParams.layerParams[0]);
            m_broadcastParams = reinterpret_cast<sw_params::BroadcastParams*>(paramContainer);
            *m_broadcastParams = sw_params::BroadcastParams();
            m_broadcastParams ->mode = m_mode;
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::BroadcastParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[1]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_broadcastParams);

#ifdef CONFIG_TARGET_SOC_3720
            m_params.kernel = reinterpret_cast<uint64_t>(sk_broadcast_3010xx_text);
#else
            m_params.kernel = reinterpret_cast<uint64_t>(PREAMBLE_FUNC(broadcast));
#endif
        }

        void initParserRunner() override
        {
            initMyriadResources();
            initDebugInfo();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
            OpTensor inBuff;
            OpTensor targetShape;
            OpTensor axesMapping;
            OpTensor outBuff;
            m_inputTensor.exportToBuffer(inBuff);
            m_targetShapeTensor.exportToBuffer(targetShape);
            m_axesMappingTensor.exportToBuffer(axesMapping);
            m_outputTensor.exportToBuffer(outBuff);

            customCppOp->addInputBuffer(inBuff, m_requiredTensorLocation);
            customCppOp->addInputBuffer(targetShape, m_requiredTensorLocation);
            customCppOp->addInputBuffer(axesMapping, m_requiredTensorLocation);
            customCppOp->addOutputBuffer(outBuff, m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.008f;
        }

        void generateInputData() override {

            std::mt19937 generator(SEED_VALUE);
            m_inputTensor.forEach(false, [this, &generator](const MemoryDims& indices) {
                // We generate the random value between -8.f and 8f and the kernel do x * rbroadcast6(x+3) / 6
                // So the minimum resolution is 2^(-7) = 0.00781f and the kernel may calculate 0 output value
                // if input value is less than this resolution. In such cases, relative difference would be 1.
                const float precisionLimitations = 0.00781f;
                float fp32Value = 0.f;
                do {
                    fp32Value = float(generator()) / generator.max() * 16.f - 8.f;
                } while (fabs(fp32Value) < precisionLimitations && fp32Value != 0.f);

                m_inputTensor.at(indices) = f32Tof16(fp32Value);
            });

            // std::vector<int32_t> targetIEShape(m_dimensionsLoop.value().targetShape);
            // std::reverse(targetIEShape.begin(), targetIEShape.end());
            // auto targetShapeIterator = targetIEShape.begin();
            // m_targetShapeTensor.forEach(false, [this, &targetShapeIterator](const MemoryDims &indices) {
            //     m_targetShapeTensor.at(indices) = *targetShapeIterator;
            //     targetShapeIterator++;
            // });

            // if (m_mode == mode.EXPLICIT) {
            //     auto axesMappingIterator = m_dimensionsLoop.value().axesMapping.begin();
            //     m_axesMappingTensor.forEach(false, [this, &axesMappingIterator](const MemoryDims &indices) {
            //         m_axesMappingTensor.at(indices) = *axesMappingIterator;
            //         axesMappingIterator++;
            //     });
            // }
        }

        void generateReferenceData() override {
            // TODO: axes mapping part???
            // std::vector<int32_t> axesMapping(m_dimensionsLoop.value().axesMapping);
            // if (axesMapping.empty()) {
            //     axesMapping.resize(m_inputTensor.ndims(), 0);
            //     std::iota(axesMapping.begin(), axesMapping.end(), 0);
            // } else {
            //     std::transform(axesMapping.begin(), axesMapping.end(), axesMapping.begin(),
            //                    [this](const int32_t &axis) { return m_outputTensor.ndims() - axis - 1; });
            //     std::reverse(axesMapping.begin(), axesMapping.end());
            // }

            // m_referenceOutputTensor.forEach(false, [this, &axesMapping](const MemoryDims &indices) {
            //     MemoryDims inCoord;
            //     for (int dim = 0; dim < m_referenceOutputTensor.ndims(); ++dim) {
            //         int inputDimValue = DefTensorDim;
            //         const auto axisMappingIt = std::find(axesMapping.begin(), axesMapping.end(), dim);
            //         if (axisMappingIt != axesMapping.end()) {
            //             const auto idxInInput = std::distance(axesMapping.begin(), axisMappingIt);
            //             inputDimValue = m_inputTensor.memoryDims().dims[idxInInput];
            //             inCoord.dims[idxInInput] = inputDimValue == 1 ? 0 : indices.dims[dim];
            //         }
            //     }
            //     m_referenceOutputTensor.at(indices) = m_inputTensor.at(inCoord);
            // });
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
                float value = f16Tof32(m_outputTensor.at(indices));
                float gt_value = f16Tof32(m_referenceOutputTensor.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= m_test_threshold);

                threshold_test_failed |= differ;

                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_outputTensor.toTensor(indices);
                    printf("DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value,
                           abs_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<SingleTest> m_testsLoop;
        ListIterator<targetShape> m_targetShape;
        ListIterator<axesMapping> m_axesMapping;

        // Additional buffer to avoid convertion back and forth
        sw_params::BroadcastParams* m_broadcastParams;
        Tensor<int32_t> m_targetShapeTensor;
        Tensor<int32_t> m_axesMappingTensor;

        BroadcastType m_mode;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppBroadcastTest)
}  // namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME,broadcast))
