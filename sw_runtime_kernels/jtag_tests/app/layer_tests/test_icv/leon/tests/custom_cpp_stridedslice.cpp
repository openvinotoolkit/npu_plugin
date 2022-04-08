
//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <custom_cpp_tests.h>
#include "mvSubspaces.h"

__attribute__((aligned(1024)))
#include "sk.single_shave_stridedslice.3720xx.text.xdat"

#include "param_stridedslice.h"

struct StridedSliceTestParams {
    std::vector<int32_t> begin;
    std::vector<int32_t> end;
    std::vector<int32_t> strides;
    std::vector<int32_t> beginMask;
    std::vector<int32_t> endMask;
    int32_t layerParams[MAX_LOCAL_PARAMS];
};

struct StridedSliceTest {
    Dims inputDims;
    Dims outputDims;
    StorageOrder storageOrder;
    StridedSliceTestParams customLayerParams;
};

struct SlicePlan {
    std::vector<int32_t> begins;
    std::vector<int32_t> ends;
    std::vector<int32_t> strides;
    std::vector<int32_t> reshape_in_shape;
    std::vector<int32_t> reshape_out_shape;
    std::vector<int32_t> reverse_axes;
};

// ngraph way of parsing StridedSlice layer
SlicePlan make_slice_plan(const std::vector<int32_t> &input_shape, const std::vector<int32_t> &begins, const std::vector<int32_t> &ends, const std::vector<int32_t> &strides,
                            const std::vector<int32_t> &lower_bounds_mask, const std::vector<int32_t> &upper_bounds_mask, const std::vector<int32_t> &new_axis_mask,
                            const std::vector<int32_t> &shrink_axis_mask, const std::vector<int32_t> &ellipsis_mask) {
    mvTensorAssert(begins.size() == ends.size());
    mvTensorAssert(ends.size() == strides.size());
    size_t num_slice_indices = begins.size();
    size_t num_real_axes = 0;
    size_t num_shrink_axes = 0;
    size_t num_new_axes = 0;
    bool ellipsis_found = false;
    // Make a pass over the original slices to make sure there is at most one
    // ellipsis, and to count up the number of shrink axes, the number of
    // "newaxis"es, and the number of "real" axes (axes that are not newaxis
    // and are not the ellipsis).
    for (size_t i = 0; i < num_slice_indices; i++) {
        if (ellipsis_mask[i]) {
            mvTensorAssert(!ellipsis_found);
            ellipsis_found = true;
        } else if (new_axis_mask[i]) {
            num_new_axes++;
        } else {
            if (shrink_axis_mask[i]) {
                num_shrink_axes++;
            }
            num_real_axes++;
        }
    }
    mvTensorAssert(num_real_axes <= input_shape.size());
    // Figure out how many axes need to be inserted when the ellipsis (which
    // may be an implicit ellipsis at the end) is expanded.
    size_t ellipsis_size = input_shape.size() - num_real_axes;
    // Initialize our slice plan.
    SlicePlan p;
    p.begins = std::vector<int32_t>(num_real_axes + ellipsis_size);
    p.ends = std::vector<int32_t>(num_real_axes + ellipsis_size);
    p.strides = std::vector<int32_t>(num_real_axes + ellipsis_size);
    p.reshape_in_shape = std::vector<int32_t>(num_real_axes + ellipsis_size);
    p.reshape_out_shape = std::vector<int32_t>(num_new_axes + num_real_axes + ellipsis_size - num_shrink_axes);
    p.reverse_axes = std::vector<int32_t>(num_real_axes + ellipsis_size);
    // Begin a maddeningly delicate loop to desugar the original slice.
    //
    // * i_in is iterating over the axes of the input shape, which are also the axes of
    //     p.reshape_in_shape.
    // * i_out is iterating over the axes of p.reshape_out_shape
    size_t i_in = 0;
    size_t i_out = 0;
    // If no actual ellipsis exists, there is an "implicit" one at the end,
    // which we will handle after the loop. So the logic is wrapped up here,
    // allowing it to be used both during and after the loop.
    auto expand_ellipsis = [&]() {
        for (size_t i = 0; i < ellipsis_size; i++) {
            p.begins[i_in] = 0;
            p.ends[i_in] = int32_t(input_shape[i_in]);
            p.strides[i_in] = 1;
            p.reshape_in_shape[i_in] = input_shape[i_in];
            p.reshape_out_shape[i_out] = input_shape[i_in];
            i_in++;
            i_out++;
        }
    };
    for (size_t i = 0; i < num_slice_indices; i++) {
        // If this is a "newaxis", then reshape_out_shape will have a 1 here,
        // but reshape_in_shape will not.
        if (new_axis_mask[i]) {
            p.reshape_out_shape[i_out] = 1;
            i_out++;
        }
        // If this is a "shrunken" axis, then reshape_in_shape will have a 1
        // here, but reshape_out_shape will not.
        else if (shrink_axis_mask[i]) {
            int32_t begin = begins[i];
            // Note that clipping is not used for "shrunken" axes: an
            // out-of-bounds index is an error.
            mvTensorAssert(begin >= -(int32_t(input_shape[i_in])) && begin < int32_t(input_shape[i_in]));
            if (begin < 0) {
                begin += int32_t(input_shape[i_in]);
            }
            p.begins[i_in] = begin;
            p.ends[i_in] = begin + 1;
            p.strides[i_in] = 1;
            p.reshape_in_shape[i_in] = 1;
            i_in++;
        }
        // If this is the ellipsis, expand it.
        else if (ellipsis_mask[i]) {
            expand_ellipsis();
        }
        // In other cases, we have a nice, ordinary (begin:end:stride) slice.
        // We need to adjust for begin/end being masked, and begin/end/stride
        // being negative or out of bounds.
        else {
            const bool is_reverse = strides[i] < 0;
            // Adjust the beginning for from-the-right indexing, and clip.
            int32_t real_begin = begins[i];
            if (lower_bounds_mask[i]) {
                real_begin = (is_reverse ? int32_t(input_shape[i_in] - 1) : 0);
            } else if (real_begin < 0) {
                real_begin += int32_t(input_shape[i_in]);
            }
            int32_t max_real_begin = int32_t(input_shape[i_in]) - (is_reverse ? 1 : 0);
            real_begin = std::max(int32_t(0), std::min(max_real_begin, real_begin));
            // Adjust the ending for from-the-right indexing, and clip.
            int32_t real_end = ends[i];
            if (upper_bounds_mask[i]) {
                real_end = (is_reverse ? -1 : int32_t(input_shape[i_in]));
            } else if (real_end < 0) {
                real_end += int32_t(input_shape[i_in]);
            }
            int32_t min_real_end = (is_reverse ? -1 : 0);
            real_end = std::max(min_real_end, std::min(int32_t(input_shape[i_in]), real_end));
            // Ensure stride is not zero, and adjust it for backwards slicing.
            mvTensorAssert(strides[i] != 0);
            int32_t real_stride = std::abs(strides[i]);
            // Adjust for reversal if needed. This isn't quite as simple as swapping begin and
            // end, due to striding; we have to adjust the end point to be the _actual_ leftmost
            // element, in cases where the stride does not evenly divide the span between begin
            // and end.
            if (is_reverse) {
                real_end += std::max(int32_t(0), real_begin - real_end - 1) % real_stride;
                std::swap(real_begin, real_end);
                real_begin++;
                real_end++;
                p.reverse_axes[i_out] = 1;
            }
            // nGraph's slice op does not like it when end < begin, so we truncate for that case
            // here.
            // if (real_end < real_begin) {
            //     real_end = real_begin;
            // }
            // Compute output dimension.
            size_t dim = (real_end <= real_begin ? 0 : ((real_end - real_begin) / real_stride));
            p.reshape_in_shape[i_in] = dim;
            p.reshape_out_shape[i_out] = dim;
            auto slice_size = real_end - real_begin;
            if (slice_size > 0 && real_stride > slice_size)
                real_stride = slice_size;
            if (real_stride == slice_size) {
                real_end = real_begin + 1;
                real_stride = 1;
            }
            // Set up the begin/end/stride.
            p.begins[i_in] = real_begin;
            p.ends[i_in] = real_end;
            p.strides[i_in] = real_stride;
            i_in++;
            i_out++;
        }
    }
    // If there was no ellipsis explicitly given, there is an implicit one at
    // the end (it might encompass zero axes, but that's fine).
    if (!ellipsis_found) {
        expand_ellipsis();
    }
    return p;
}

namespace ICV_TESTS_NAMESPACE(ICV_TESTS_PASTE2(ICV_TEST_SUITE_NAME, StridedSlice)) {
    const bool save_to_file = false;
    // clang-format off
    static std::initializer_list<StridedSliceTest> stridedslice_test_list{
            // all the tensor sizes are in a logical layout (from outer to inner dimension)
            //  N   C   H    W                                                           begins          ends            strides       beginsMask       endsMask
            {{1, 16, 16, 16 }, {1, 16, 16, 16 }, FULL_ORDER, {{ 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, {sw_params::Location::NN_CMX}}},
            {{1, 16, 16, 16 }, {1, 16, 8, 16 }, FULL_ORDER, {{ 0, 0, 0, 0 }, { 0, 0, 0, 0 }, { 1, 1, 2, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 }, {sw_params::Location::NN_CMX}}},
            {{5, 16, 16, 16 }, {2, 1, 2, 2 }, FULL_ORDER, {{ 1, 4, 5, 10 }, { 0, 0, 0, 0 }, { 2, 7, 5, 3 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 }, {sw_params::Location::NN_CMX}}},
            {{5, 16, 16, 16 }, {1, 1, 2, 3 }, FULL_ORDER, {{ 2, 8, 3, 7 }, { 0, 0, 0, 0 }, { 2, 7, 5, 3 }, { 0, 0, 0, 0 }, { 1, 1, 1, 1 }, {sw_params::Location::NN_CMX}}}
    };

    class CustomCppStridedSliceTest : public CustomCppTests<fp16, StridedSliceTest> {
    public:
        explicit CustomCppStridedSliceTest(): m_testsLoop(stridedslice_test_list, "test") {
        }
        virtual ~CustomCppStridedSliceTest() {
        }
    protected:
        const char* suiteName() const override {
            return "CustomCppStridedSliceTest";
        }
        
        void userLoops() override {
            addLoop(m_testsLoop);
        }
        void initData() override {
            sw_params::BaseKernelParams emptyParamData;
            m_params = {nullptr, emptyParamData, 0, 0xFFFFFFFF, 0, MAX_LOCAL_PARAMS};
            CustomCppTests<fp16, StridedSliceTest>::initData();
            const StridedSliceTest* test = m_currentTest;
            const auto &params = m_testsLoop.value();
            m_inDims = params.inputDims;
            m_sliceParams = params.customLayerParams;
            const StorageOrder& storageOrder = m_currentTest->storageOrder;
            // Check input params
            const auto beginMaskSize = params.customLayerParams.beginMask.size();
            const auto endMaskSize = params.customLayerParams.endMask.size();
            mvTensorAssert(beginMaskSize == endMaskSize, "sizes of the masks should be equal");
            auto empty = std::vector<int32_t>(m_sliceParams.beginMask.size());
            auto plan =
                make_slice_plan(m_inDims, m_sliceParams.begin, m_sliceParams.end, m_sliceParams.strides, m_sliceParams.beginMask,
                                m_sliceParams.endMask, empty, empty, empty);
            // We do the "hard" work of slicing on compiler side
            // Kernel only does a copy
            m_act_params.begin = plan.begins;
            m_act_params.end = plan.ends;
            m_act_params.strides = plan.strides;
            m_outDims = plan.reshape_out_shape;
            std::reverse(m_inDims.begin(), m_inDims.end());
            std::reverse(m_outDims.begin(), m_outDims.end());
            
            MemoryDims inputDims(m_inDims.data(), m_inDims.size());
            MemoryDims outputDims(m_outDims.data(), m_outDims.size());
            m_inputTensor.init(maskOrder(storageOrder, m_inDims.size()), inputDims, inputDims);
            m_outputTensor.init(maskOrder(storageOrder, m_outDims.size()), outputDims, outputDims);
            m_referenceInput.init(maskOrder(storageOrder, m_inDims.size()), inputDims, inputDims);
            allocBuffer(m_inputTensor);
            allocBuffer(m_referenceInput);
            allocBuffer(m_outputTensor);
            m_stridedsliceParams = reinterpret_cast<sw_params::StridedSliceParams*>(paramContainer);
            *m_stridedsliceParams = sw_params::StridedSliceParams();
            for(uint32_t i=0; i < m_act_params.begin.size(); i++)
            {
                m_stridedsliceParams->begins[i] = m_act_params.begin[i];
                m_stridedsliceParams->ends[i] = m_act_params.end[i];
                m_stridedsliceParams->strides[i] = m_act_params.strides[i];
            }             
            m_params.paramData = reinterpret_cast<uint32_t*>(paramContainer);
            m_params.paramDataLen = sizeof(sw_params::StridedSliceParams);
            m_requiredTensorLocation = static_cast<sw_params::Location>(test->customLayerParams.layerParams[0]);
            m_params.baseParamData = sw_params::stridedSliceParamsToBaseKernelParams(m_stridedsliceParams);
            m_params.kernel = reinterpret_cast<uint64_t>(sk_single_shave_stridedslice_3720xx_text);
        }
        void initTestCase() override {
            m_currentTest = &m_testsLoop.value();
            m_test_threshold = 0.001f;
        }

        void initParserRunner() override {
            initMyriadResources();
            static_assert(std::is_base_of<Op, CustomCpp>());
            CustomCpp* customCppOp = static_cast<CustomCpp*>(m_op);
            OpTensor inputBuff;
            OpTensor outputBuff;
            m_inputTensor.exportToBuffer(inputBuff);
            m_outputTensor.exportToBuffer(outputBuff);
            customCppOp->addInputBuffer(inputBuff, m_requiredTensorLocation);
            customCppOp->addOutputBuffer(outputBuff, m_requiredTensorLocation);
            customCppOp->ops = *getParams();
        }

        void generateReferenceData() override {}
        
        void generateInputData() override {
            int i = 100;
            m_inputTensor.forEach(false, [&](const MemoryDims& indices) {
                if (i > 200.0)
                    i = 100;
                float val = (float)(i++);
                m_inputTensor.at(indices) = f32Tof16(val);
                m_referenceInput.at(indices) = f32Tof16(val);
            });
        }

        virtual bool checkResult() override {
            m_outputTensor.confirmBufferData();
            // save output data
            if (save_to_file) {
                saveMemoryToFile(reinterpret_cast<u32>(m_outputTensor.buffer()), m_outputTensor.bufferSize(),
                                    "outMyriad.bin");
            }
            bool threshold_test_failed = false;
            m_outputTensor.forEach(false, [&](const MemoryDims &indices) {
                int offset = 0;
                for (size_t i = 0; i < m_act_params.begin.size(); i++) {
                    int idx = m_act_params.begin.size() - i - 1;
                    offset += (m_act_params.begin[idx] + indices.dims[i] * m_act_params.strides[idx]) *
                                m_referenceInput.memorySteps().dims[i];
                }
                float gt_value = f16Tof32(m_referenceInput.data()[offset]);
                float value = f16Tof32(m_outputTensor.at(indices, true));
                float abs_diff = fabs(value - gt_value);
                bool differ = !bool(abs_diff <= m_test_threshold);
                threshold_test_failed |= differ;
                if (differ && GlobalData::doPrintDiffs) {
                    char indices_str[64];
                    printf("DIFF [%s] val: %f ref: %f\n", m_outputTensor.indicesToString(indices, indices_str),
                            value, gt_value);
                }
            });
            return !threshold_test_failed;
        }
    private:
        ListIterator<StridedSliceTest> m_testsLoop;
        sw_params::StridedSliceParams* m_stridedsliceParams;
        Tensor<fp16> m_inputTensor;
        Tensor<fp16> m_referenceInput;
        Tensor<fp16> m_outputTensor;
        std::vector<int32_t> m_inDims;
        std::vector<int32_t> m_outDims;
        StridedSliceTestParams m_sliceParams;
        StridedSliceTestParams m_act_params;
    };
ICV_TESTS_REGISTER_SUITE(CustomCppStridedSliceTest)
}
