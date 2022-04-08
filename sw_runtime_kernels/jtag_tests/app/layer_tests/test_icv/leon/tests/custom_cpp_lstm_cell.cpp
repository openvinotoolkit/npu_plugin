//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include <random>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.lstm_cell.3720xx.text.xdat"

#include "param_lstm_cell.h"

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, LSTMCell)) {
    struct LSTMCellTest {
        Dims inputDims;
        Dims outputDims;
        int input_size;
        int state_size;
        int nBatches;
        int nCells;
        int outputsNumber;
        bool RNNForward;
        StorageOrder storageOrder;
        CustomParams customLayerParams;
    };
    const float test_threshold = 0.06f;
    const int n_gates = 4;
    const int gate_map[n_gates] = {0, 1, 3, 2};
    const bool saveToFile = false;

    static constexpr std::initializer_list<LSTMCellTest> lstm_cell_test_list{
            //  LSTM Cell test cases
            {{1, 1, 1}, {1, 1, 1}, 6, 4, 1, 1, 2, true, FULL_ORDER, {sw_params::Location::NN_CMX /*mem type*/}},
            {{1, 1, 1}, {1, 1, 1}, 12, 8, 1, 1, 2, true, FULL_ORDER, {sw_params::Location::NN_CMX /*mem type*/}},
            {{1, 1, 1}, {1, 1, 1}, 26, 10, 1, 1, 2, true, FULL_ORDER, {sw_params::Location::NN_CMX /*mem type*/}},
            {{1, 1, 1}, {1, 1, 1}, 100, 2, 1, 1, 2, true, FULL_ORDER, {sw_params::Location::NN_CMX /*mem type*/}},
    };

    static fp16& at(fp16 * a, int i, int j, int k, int stride0, int stride1) {
        return *(i * stride1 + j * stride0 + k + a);
    }
    static fp16& at(fp16 * a, int i, int j, int stride) {
        return *(i * stride + j + a);
    }
    // float a[m][k], float b[k][n], float c[m][n];
    // c = a * b;
    static void gemm(int m, int n, int k, fp16* a, int stride_a, fp16* b, int stride_b, fp16* c, int stride_c,
                     fp16 beta) {
        for (int im = 0; im < m; im++) {
            for (int in = 0; in < n; in++) {
                fp16 c_elem =
                        (beta == (fp16)0.) ? (fp16)0. : f32Tof16(f16Tof32(at(c, im, in, stride_c)) * f16Tof32(beta));
                for (int ik = 0; ik < k; ik++) {
                    fp16 a_elem = at(a, im, ik, stride_a);
                    fp16 b_elem = at(b, ik, in, stride_b);
                    c_elem = f32Tof16(f16Tof32(a_elem) * f16Tof32(b_elem) + f16Tof32(c_elem));
                }
                at(c, im, in, stride_c) = c_elem;
            }
        }
    }
    static float logistic(float x) {
        return 1.0f / (1.0f + expf(-x));
    }
    static void lstm_activation(int dic, int n_gates, int batch, fp16* a) {
        for (int ib = 0; ib < batch; ib++) {
            for (int ig = 0; ig < 3; ig++) {
                for (int ih = 0; ih < dic; ih++) {
                    *(a + ih + ig * dic + ib * dic * n_gates) =
                            f32Tof16(logistic(f16Tof32(*(a + ih + ig * dic + ib * dic * n_gates))));
                }
            }
            int ig = 3;
            for (int j = 0; j < dic; j++) {
                *(a + j + ig * dic + ib * dic * n_gates) =
                        f32Tof16(tanhf(f16Tof32(*(a + j + ig * dic + ib * dic * n_gates))));
            }
        }
    }

    // src_layer[input_size]
    // src_iter_h[state_size]
    // src_iter_c[state_size]
    // weights_layer[n_gates * state_size][input_size]
    // weights_iter_h[n_gates * state_size][state_size]
    // bias[n_gates][state_size]
    // h_dst[state_size]
    // c_dst[state_size]
    void lstm_cell_ref(int input_size, int state_size,

                       // weights
                       fp16* weights_layer, fp16* weights_iter_h, fp16* bias,

                       // input
                       fp16* src_layer, fp16* src_iter_h, fp16* src_iter_c,

                       int outputs_number,

                       // output
                       fp16* h_dst, fp16* c_dst, fp16* l_h_dst,

                       fp16* gates) {
        const int ohf = 0;
        const int ohi = 1;
        const int oho = 2;
        const int ohc = 3;

        /* gates = src_layer * weights_layer */
        gemm(1, n_gates * state_size, input_size, src_layer, input_size, weights_layer, n_gates * state_size, gates,
             n_gates * state_size, f32Tof16(0.0f));

        /* gates += src_iter_h * weights_iter_h */
        gemm(1, n_gates * state_size, state_size, src_iter_h, state_size, weights_iter_h, n_gates * state_size, gates,
             n_gates * state_size, f32Tof16(1.0f));

        // add bias
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < n_gates; j++) {
                for (int k = 0; k < state_size; k++) {
                    *(gates + i * n_gates * state_size + j * state_size + k) =
                            f32Tof16(f16Tof32(*(gates + i * n_gates * state_size + j * state_size + k)) +
                                     f16Tof32(*(bias + j * state_size + k)));
                }
            }
        }

        // run the eltwise
        lstm_activation(state_size, n_gates, 1, gates);

        // compute C_t_l and H_t_l
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < state_size; j++) {
                float tmp = f16Tof32(at(gates, i, ohf, j, state_size, state_size * n_gates)) *
                                    f16Tof32(at(src_iter_c, i, j, state_size)) +
                            f16Tof32(at(gates, i, ohi, j, state_size, state_size * n_gates)) *
                                    f16Tof32(at(gates, i, ohc, j, state_size, state_size * n_gates));
                at(c_dst, i, j, state_size) = f32Tof16(tmp);
                at(h_dst, i, j, state_size) =
                        f32Tof16(f16Tof32(at(gates, i, oho, j, state_size, state_size * n_gates)) * tanhf(tmp));
                if (outputs_number == 3 && l_h_dst)
                    at(l_h_dst, i, j, state_size) =
                            f32Tof16(f16Tof32(at(gates, i, oho, j, state_size, state_size * n_gates)) * tanhf(tmp));
            }
        }
    }

    /* psrc[m][n] -> pdst[n][m] */
    static void matrix_copy_transpose(const fp16* psrc, fp16* pdst, int m, int n) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                pdst[j * m + i] = psrc[i * n + j];
            }
        }
    }

    class CustomCppLSTMCellTest : public CustomCppTests<fp16, LSTMCellTest> {
    public:
        explicit CustomCppLSTMCellTest(): m_testsLoop(lstm_cell_test_list, "test") {
        }
        virtual ~CustomCppLSTMCellTest() {
        }

    protected:
        const char* suiteName() const override {
            return "CustomCppLSTMCellTest";
        }
        void userLoops() override {
            addLoop(m_testsLoop);
        }

        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};

            initTestCase();
            m_input_size = m_currentTest->input_size;
            m_state_size = m_currentTest->state_size;
            const int nCells = m_currentTest->nCells;
            const int nBatches = m_currentTest->nBatches;
            outputs_number = m_currentTest->outputsNumber;

            m_h_dst.init(orderXYZ, TensorDims(1, m_state_size * nCells * nBatches, 1, 1), TensorAlign(0, 0, 0, 0));
            m_c_dst.init(orderXYZ, TensorDims(1, m_state_size * nBatches, 1, 1), TensorAlign(0, 0, 0, 0));
            m_last_h_dst.init(orderXYZ, TensorDims(1, m_state_size * nCells * nBatches, 1, 1), TensorAlign(0, 0, 0, 0));

            m_c_dst_ref.init(orderXYZ, TensorDims(1, m_state_size * nBatches, 1, 1), TensorAlign(0, 0, 0, 0));
            m_h_dst_ref.init(orderXYZ, TensorDims(1, m_state_size * nCells * nBatches, 1, 1), TensorAlign(0, 0, 0, 0));
            m_last_h_dst_ref.init(orderXYZ, TensorDims(1, m_state_size * nCells * nBatches, 1, 1),
                                  TensorAlign(0, 0, 0, 0));

            m_src_layer.init(orderXYZ, TensorDims(1, m_input_size * nCells * nBatches, 1, 1), TensorAlign(0, 0, 0, 0));
            m_src_iter_h.init(orderXYZ, TensorDims(1, m_state_size * nBatches, 1, 1), TensorAlign(0, 0, 0, 0));
            m_src_iter_c.init(orderXYZ, TensorDims(1, m_state_size * nBatches, 1, 1), TensorAlign(0, 0, 0, 0));

            m_weights_layer.init(orderXYZ, TensorDims(1, n_gates * m_state_size * (m_input_size + m_state_size), 1, 1),
                                 TensorAlign(0, 0, 0, 0));
            m_weights_layer_inv.init(orderXYZ,
                                     TensorDims(1, n_gates * m_state_size * (m_input_size + m_state_size), 1, 1),
                                     TensorAlign(0, 0, 0, 0));
            m_weights_layer_hidden.init(orderXYZ, TensorDims(1, n_gates * m_state_size * m_state_size, 1, 1),
                                        TensorAlign(0, 0, 0, 0));
            m_biases_layer.init(orderXYZ, TensorDims(1, n_gates * m_state_size, 1, 1), TensorAlign(0, 0, 0, 0));
            m_biases_layer_inv.init(orderXYZ, TensorDims(1, n_gates * m_state_size, 1, 1), TensorAlign(0, 0, 0, 0));

            m_weights_layer_repacked.init(orderXYZ, TensorDims(1, m_input_size * n_gates * m_state_size, 1, 1),
                                          TensorAlign(0, 0, 0, 0));
            m_weights_iter_h_repacked.init(orderXYZ, TensorDims(1, m_state_size * n_gates * m_state_size, 1, 1),
                                           TensorAlign(0, 0, 0, 0));
            m_gates.init(orderXYZ, TensorDims(1, n_gates * m_state_size, 1, 1), TensorAlign(0, 0, 0, 0));

            allocBuffer(m_src_layer);
            allocBuffer(m_src_iter_h);
            allocBuffer(m_src_iter_c);

            allocBuffer(m_weights_layer);
            allocBuffer(m_weights_layer_inv);
            allocBuffer(m_weights_layer_hidden);
            allocBuffer(m_biases_layer);
            allocBuffer(m_biases_layer_inv);

            allocBuffer(m_h_dst);
            allocBuffer(m_c_dst);
            allocBuffer(m_last_h_dst);

            allocBuffer(m_h_dst_ref);
            allocBuffer(m_c_dst_ref);
            allocBuffer(m_last_h_dst_ref);

            allocBuffer(m_weights_layer_repacked);
            allocBuffer(m_weights_iter_h_repacked);
            allocBuffer(m_gates);

            m_lstmCellParams = reinterpret_cast<sw_params::LSTMCellParams*>(paramContainer);
            *m_lstmCellParams = sw_params::LSTMCellParams();
            m_lstmCellParams->RNNForward = static_cast<int64_t>(m_currentTest->RNNForward);
            m_lstmCellParams->nCells = static_cast<int64_t>(m_currentTest->nCells);
            m_lstmCellParams->nBatches = static_cast<int64_t>(m_currentTest->nBatches);
            m_lstmCellParams->outputsNumber = static_cast<int64_t>(m_currentTest->outputsNumber);
            m_lstmCellParams->useCellState = static_cast<int64_t>(1);

            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::LSTMCellParams);
            m_requiredTensorLocation =
                    static_cast<sw_params::Location>(m_currentTest->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::ToBaseKernelParams(m_lstmCellParams);
            m_params.kernel = reinterpret_cast<uint32_t>(sk_lstm_cell_3720xx_text);
        }

        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.1f;
        }

        void initParserRunner() override {
            initMyriadResources();

            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);

            OpTensor inputData;
            OpTensor initialHiddenState;
            OpTensor initialCellState;
            OpTensor weights;
            OpTensor weightsHidden;
            OpTensor biases;
            OpTensor outputHiddenState;
            OpTensor outputCellState;

            m_src_layer.exportToBuffer(inputData);
            customCppOp->addInputBuffer(inputData, m_requiredTensorLocation);

            m_src_iter_h.exportToBuffer(initialHiddenState);
            customCppOp->addInputBuffer(initialHiddenState, m_requiredTensorLocation);

            m_src_iter_c.exportToBuffer(initialCellState);
            customCppOp->addInputBuffer(initialCellState, m_requiredTensorLocation);

            m_weights_layer.exportToBuffer(weights);
            customCppOp->addInputBuffer(weights, m_requiredTensorLocation);

            m_weights_layer_hidden.exportToBuffer(weightsHidden);
            customCppOp->addInputBuffer(weightsHidden, m_requiredTensorLocation);

            m_biases_layer.exportToBuffer(biases);
            customCppOp->addInputBuffer(biases, m_requiredTensorLocation);

            m_h_dst.exportToBuffer(outputHiddenState);
            customCppOp->addOutputBuffer(outputHiddenState, m_requiredTensorLocation);

            m_c_dst.exportToBuffer(outputCellState);
            customCppOp->addOutputBuffer(outputCellState, m_requiredTensorLocation);

            customCppOp->ops = *getParams();
        }

        void generateInputData() override {
            /* source tensor generating */
            m_src_layer.forEach(false, [&](const MemoryDims& indices) {
                float tmp = (((float)(rand() % m_input_size)) / m_input_size - 0.5f) * .1f;
                m_src_layer.at(indices) = f32Tof16(tmp);
            });
            m_src_iter_h.forEach(false, [&](const MemoryDims& indices) {
                float tmp = (((float)(rand() % m_state_size)) / m_state_size - 0.5f) * .2f;
                m_src_iter_h.at(indices) = f32Tof16(tmp);
            });
            m_src_iter_c.forEach(false, [&](const MemoryDims& indices) {
                float tmp = (((float)(rand() % m_state_size)) / m_state_size - 0.5f) * .3f;
                m_src_iter_c.at(indices) = f32Tof16(tmp);
            });

            /* weights tensor generating */
            fp16* pweights = m_weights_layer.data();
            fp16* pweights_iter_h = pweights + n_gates * m_state_size * m_input_size;
            fp16* pweights_hidden = m_weights_layer_hidden.data();
            fp16* pbias = m_biases_layer.data();
            for (int j = 0; j < n_gates * m_state_size; j++) {
                for (int i = 0; i < m_input_size; i++) {
                    pweights[(m_input_size)*j + i] =
                            f32Tof16((((float)(rand() % m_input_size)) / m_input_size - 0.5f) * 0.01);
                }
                for (int i = 0; i < m_state_size; i++) {
                    pweights_hidden[(m_state_size)*j + i] = pweights_iter_h[(m_state_size)*j + i] =
                            f32Tof16((((float)(rand() % m_state_size)) / m_state_size - 0.5f) * 0.05f);
                }
            }
            int num_bias = n_gates * m_state_size;
            for (int i = 0; i < num_bias; i++) {
                pbias[i] = f32Tof16((float)(rand() % (num_bias)) / num_bias - 0.5f);
            }
        }

        void generateReferenceData() override {
            int input_size = m_currentTest->input_size;
            int state_size = m_currentTest->state_size;
            int nCells = m_currentTest->nCells;
            int nBatches = m_currentTest->nBatches;
            bool RNNForward = m_currentTest->RNNForward;

            fp16* pinput0 = (fp16*)m_src_layer.data();
            fp16* pinput1 = (fp16*)m_src_iter_h.data();
            fp16* pinput2 = (fp16*)m_src_iter_c.data();

            fp16* poutput0 = (fp16*)m_h_dst_ref.data();
            fp16* poutput1 = (fp16*)m_c_dst_ref.data();
            fp16* poutput2 = (fp16*)m_last_h_dst_ref.data();

            fp16* pweights_layer = (fp16*)m_weights_layer.data();
            fp16* pweights_iter_h = pweights_layer + n_gates * state_size * input_size;

            fp16* pweights_layer_inv = (fp16*)m_weights_layer_inv.data();
            fp16* pweights_iter_h_inv = pweights_layer_inv + n_gates * state_size * input_size;

            fp16* pbias = (fp16*)m_biases_layer.data();
            fp16* pbias_inv = (fp16*)m_biases_layer_inv.data();

            // gates repacking
            {
                for (int g = 0; g < n_gates; g++) {
                    int stride = state_size * input_size;
                    for (int i = 0; i < stride; i++) {
                        pweights_layer_inv[g * stride + i] = pweights_layer[gate_map[g] * stride + i];
                    }
                }
                for (int g = 0; g < n_gates; g++) {
                    int stride = state_size * state_size;
                    for (int i = 0; i < stride; i++) {
                        pweights_iter_h_inv[g * stride + i] = pweights_iter_h[gate_map[g] * stride + i];
                    }
                }
                for (int g = 0; g < n_gates; g++) {
                    int stride = state_size;
                    for (int i = 0; i < stride; i++) {
                        pbias_inv[g * stride + i] = pbias[gate_map[g] * stride + i];
                    }
                }
            }
            /* weights repacking */
            matrix_copy_transpose(pweights_layer_inv, m_weights_layer_repacked.data(), n_gates * state_size,
                                  input_size);
            matrix_copy_transpose(pweights_iter_h_inv, m_weights_iter_h_repacked.data(), n_gates * state_size,
                                  state_size);

            int cellStart = RNNForward ? 0 : (nCells - 1);
            int cellStride = RNNForward ? 1 : (-1);

            for (int b = 0; b < nBatches; b++) {
                for (int c = 0; c < nCells; c++) {
                    int cellInd = cellStart + cellStride * c;
                    int cellPrevInd = cellStart + cellStride * (c - 1);

                    lstm_cell_ref(
                            input_size, state_size,

                            // weights
                            m_weights_layer_repacked.data(), m_weights_iter_h_repacked.data(), pbias_inv,

                            // inputs
                            pinput0 + input_size * cellInd + input_size * nCells * b,
                            (c == 0) ? (pinput1 + state_size * b)
                                     : (poutput0 + state_size * cellPrevInd + state_size * nCells * b),
                            (c == 0) ? (pinput2 + state_size * b) : (poutput1),

                            outputs_number,

                            // outputs
                            poutput0 + state_size * cellInd + state_size * nCells * b, poutput1,
                            (c == nCells - 1) ? poutput2 + state_size * cellInd + state_size * nCells * b : nullptr,

                            m_gates.data());
                }
            }
        }

        void resetOutputData() override {
            resetTensorBuffer(m_h_dst);
            resetTensorBuffer(m_c_dst);
        }

        virtual bool checkResult() override {
            m_h_dst.confirmBufferData();
            m_c_dst.confirmBufferData();

            // save output data
            if (saveToFile) {
                saveMemoryToFile(reinterpret_cast<u32>(m_h_dst.buffer()), m_h_dst.bufferSize(), "outMyriad.bin");
            }

            bool threshold_test_failed = false;
            m_h_dst.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_h_dst.at(indices));
                float gt_value = f16Tof32(m_h_dst_ref.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= test_threshold);

                threshold_test_failed |= differ;
                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_h_dst.toTensor(indices);
                    printf("M_H_DST DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value,
                           abs_diff);
                }
            });
            m_c_dst.forEach(false, [&](const MemoryDims& indices) {
                float value = f16Tof32(m_c_dst.at(indices));
                float gt_value = f16Tof32(m_c_dst_ref.at(indices));
                float abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= test_threshold);

                threshold_test_failed |= differ;
                if (differ && GlobalData::doPrintDiffs) {
                    const TensorDims ti = m_c_dst.toTensor(indices);
                    printf("M_C_DST DIFF HWC [%d:%d:%d] %f %f %f\n", ti.height, ti.width, ti.channels, value, gt_value,
                           abs_diff);
                }
            });

            return !threshold_test_failed;
        }

    private:
        ListIterator<LSTMCellTest> m_testsLoop;

        Tensor<fp16> m_src_layer;
        Tensor<fp16> m_src_iter_h;
        Tensor<fp16> m_src_iter_c;
        Tensor<fp16> m_weights_layer;
        Tensor<fp16> m_weights_layer_hidden;
        Tensor<fp16> m_biases_layer;
        Tensor<fp16> m_biases_layer_inv;

        Tensor<fp16> m_h_dst;
        Tensor<fp16> m_c_dst;
        Tensor<fp16> m_last_h_dst;

        Tensor<fp16> m_h_dst_ref;
        Tensor<fp16> m_c_dst_ref;
        Tensor<fp16> m_last_h_dst_ref;

        /* temporary tensors for reference version */
        Tensor<fp16> m_weights_layer_repacked;
        Tensor<fp16> m_weights_iter_h_repacked;
        Tensor<fp16> m_weights_layer_inv;
        Tensor<fp16> m_weights_iter_h_inv;

        Tensor<fp16> m_gates;

        int m_input_size;
        int m_state_size;
        int outputs_number;

        sw_params::LSTMCellParams* m_lstmCellParams;
    };

    ICV_TESTS_REGISTER_SUITE(CustomCppLSTMCellTest)
}  // namespace
