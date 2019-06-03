// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <cmath>

#define ERROR_BOUND (.1f)

using namespace InferenceEngine;

struct argmax_test_params {
    int32_t top_k;
    int32_t out_max_val;
    int32_t axis;
    tensor_test_params in;
    tensor_test_params out;
    friend std::ostream& operator<<(std::ostream& os, argmax_test_params const& tst)
    {
        return os << tst.in
                  << ", top_k=" << tst.top_k
                  << ", out_max_val=" << tst.out_max_val
                  << ", axis=" << tst.axis;
    };
};


// helper to access 1 element of blob, indices are in order of {N,C,H,W} for any layout
template<class DataType>
DataType& Blob_at(const InferenceEngine::Blob::Ptr p_blob, const SizeVector& indices) {
    auto ptr = reinterpret_cast<uint8_t *>((void*)p_blob->buffer());
    return *reinterpret_cast<DataType*>(ptr + p_blob->getTensorDesc().offset(indices)*p_blob->element_size());
}

// helper to iterate over subspace of a blob
template<class Action>
void Blob_forEach_NCHW(const InferenceEngine::Blob::Ptr p_blob, uint32_t subspace_mask, const SizeVector & start, Action func) {
    SizeVector dims = p_blob->getTensorDesc().getDims();

    auto N = dims.size();
    SizeVector indices = start;
    //enumerate part of the tensor
    int carry_in = 0;
    int serial_num = 0;

    while(indices.size() < N)
        indices.push_back(0);

    // clear the part belong to current subspace
    for (int i = 0; i < N; ++i)
        if(subspace_mask & (1 << i))
            indices[i] = 0;

    for(;;) {
        //traversal order is  W,H,C,N or 3 2 1 0
        for (int i = N-1; i >= 0 && carry_in; --i){
            // skip the coordinates out of current subspace
            // leave them unchanged;
            if(subspace_mask & (1 << i)) {
                indices[i] += carry_in;
                if(indices[i] >= dims[i])
                    indices[i] = 0; //keep carry in
                else
                    carry_in = 0; // reset carry in
            }
        }
        // overflow means iteration is done
        if(carry_in) break;

        func(indices, serial_num);
        serial_num++;

        // increase by 1
        carry_in = 1;
    }
}

#define GEN_REF_ARGMAX_DEBUG 0

static void gen_ref_argmax(const InferenceEngine::Blob::Ptr src,
                          InferenceEngine::Blob::Ptr dst,
                          argmax_test_params p){
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    uint16_t *srcData = src->buffer();
    uint16_t *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);

    auto & dims = src->getTensorDesc().getDims();

    auto N = dims.size();
    uint32_t dims_mask_channels = 0;
    uint32_t dims_mask_elements = 0;
    int channels_per_element = 1;

#if GEN_REF_ARGMAX_DEBUG == 1
    {
        printf("=========== gen_ref_argmax ==================\r\n");
        std::cout << "axis, top_k, out_max_val = " << p.axis << "," << p.top_k << "," << p.out_max_val << std::endl;
        auto layout = src->getTensorDesc().getLayout();
        if(layout == NCHW){
            printf("layout is NCHW\r\n");
        }
        else
        if(layout == NHWC){
            printf("layout is NHWC\r\n");
        }
        else
            ASSERT_NE(0, 0);
        std::cout << "getPrecision =" << src->getTensorDesc().getPrecision() << std::endl;
        std::cout << "dims    =" << dims[0] << "," << dims[1] << "," << dims[2] << "," << dims[3] << std::endl;
        auto probe_stride = [&](int d){
            SizeVector idx{0,0,0,0};
            auto p1 = (&Blob_at<uint8_t>(src, idx));
            idx[d]++;
            auto p2 = (&Blob_at<uint8_t>(src, idx));
            return reinterpret_cast<uintptr_t>(p2) - reinterpret_cast<uintptr_t>(p1);
        };
        std::cout << "probe_stride =" << probe_stride(0) << ","<< probe_stride(1) << ","<< probe_stride(2) << ","<< probe_stride(3) << std::endl;
        printf("=========== [END] ==================\r\n");
    }
#endif

    if (p.axis >= 0) {
        for(int i=0; i<N; i++){
            if(i == p.axis){
                dims_mask_channels |= 1 << i;
                channels_per_element *= dims[i];
            }
            else
                dims_mask_elements |= 1 << i;
        }
    } else {
        for(int i=0;i<N;i++){
            if(i == 0)
                dims_mask_elements |= 1 << i;
            else{
                dims_mask_channels |= 1 << i;
                channels_per_element *= dims[i];
            }
        }
    }

    std::vector<std::pair<float, int> > src_vector(channels_per_element);
    const int valid_top_k = std::min(channels_per_element, p.top_k);

    Blob_forEach_NCHW(src, dims_mask_elements, SizeVector(N, 0), [&](const SizeVector & index_element, int serial_i){
        // enumerate channel subspace and collect all channels
        Blob_forEach_NCHW(src, dims_mask_channels, index_element, [&](const SizeVector & index, int serial_j){
            // combine channels & element
            auto data = PrecisionUtils::f16tof32(Blob_at<ie_fp16>(src, index));
            src_vector[serial_j] = std::make_pair(data, serial_j);
        });

        //sort
        std::partial_sort(src_vector.begin(), src_vector.begin() + valid_top_k,
                          src_vector.end(), std::greater<std::pair<float, int> >());

        //store result
        if (p.axis >= 0){
            // output is same dimension & layout as input unless channels dimension reduced to topk
            SizeVector out_index = index_element;
            for (int j = 0; j < valid_top_k; ++j) {
                out_index[p.axis] = j;
                if (p.out_max_val)
                    Blob_at<ie_fp16>(dst, out_index) = PrecisionUtils::f32tof16(src_vector[j].first);
                else
                    Blob_at<ie_fp16>(dst, out_index) = PrecisionUtils::f32tof16(src_vector[j].second);
            }
        } else {
            // output is Nx[1|2]xTopK
            SizeVector out_index(3, 0);
            out_index[0] = index_element[0];
            for (int j = 0; j < valid_top_k; ++j) {
                out_index[2] = j;
                out_index[1] = 0;
                Blob_at<ie_fp16>(dst, out_index) = PrecisionUtils::f32tof16(src_vector[j].second);
                if (p.out_max_val){
                    out_index[1] = 1;
                    Blob_at<ie_fp16>(dst, out_index) = PrecisionUtils::f32tof16(src_vector[j].first);
                }
            }
        }
    });
}

class myriadLayersTestsArgMax_nightly: public myriadLayersTests_nightly,
                           public testing::WithParamInterface<argmax_test_params> {
    virtual void SetUp() {
        myriadLayersTests_nightly::SetUp();
        //disable batch
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = "NO";
    }
};

TEST_P(myriadLayersTestsArgMax_nightly, TestsArgMax) {
    auto p = ::testing::WithParamInterface<argmax_test_params>::GetParam();

    std::map<std::string, std::string> params;
    params["top_k"] = std::to_string(p.top_k);
    params["out_max_val"] = std::to_string(p.out_max_val);
    params["axis"] = std::to_string(p.axis);

    SetInputTensor(p.in);
    SetOutputTensor(p.out);
    NetworkInit("ArgMax",
                &params,
                0,
                0,
                nullptr,
                InferenceEngine::Precision::FP16, // output precision
                InferenceEngine::Precision::FP16  // input precision
    );
    /* input data preparation */
    SetFirstInputToRange(-100.f, 100.f);
    ASSERT_TRUE(Infer());

    /* output check */
    auto outputBlob =_outputMap[_outputsInfo.begin()->first];
    auto inputBlob  = _inputMap[_inputsInfo.begin()->first];

    gen_ref_argmax(inputBlob, _refBlob, p);

    Compare(outputBlob, _refBlob, ERROR_BOUND);
}

INSTANTIATE_TEST_CASE_P(
    accuracy, myriadLayersTestsArgMax_nightly,
    ::testing::Values(
        argmax_test_params{1,1,1, {1,3,11,21},{1,1,11,21}},
        argmax_test_params{1,0,1, {1,3,11,21},{1,1,11,21}},
        argmax_test_params{2,1,1, {1,3,11,21},{1,2,11,21}},
        argmax_test_params{2,0,1, {1,3,11,21},{1,2,11,21}},
        argmax_test_params{16,1,-1, {10,3,11,21},{10,2,16,1}},
        argmax_test_params{16,0,-1,{10,3,11,21},{10,1,16,1}}));
