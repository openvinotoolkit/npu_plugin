// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gapi_test_computations.hpp>
#include <opencv2/gapi.hpp>
#include <kmb_preproc_gapi_kernels.hpp>
#include <kmb_preproc_gapi_kernels_sipp.hpp>

#define CV_MAT_CHANNELS(flags) (((flags) >> CV_CN_SHIFT) + 1)

struct ComputationBase::Priv
{
    cv::GComputation m_c;
    std::vector<cv::gapi::own::Mat> m_v_in;
    std::vector<cv::gapi::own::Mat> m_v_out;
};

ComputationBase::ComputationBase(Priv *priv)
    : m_priv(priv)
{}

void ComputationBase::warmUp()
{
    m_priv->m_c.apply(m_priv->m_v_in, m_priv->m_v_out,
                      cv::compile_args(InferenceEngine::gapi::preproc::sipp::kernels()));
}

void ComputationBase::apply()
{
    m_priv->m_c.apply(m_priv->m_v_in, m_priv->m_v_out);
}

namespace
{
cv::gapi::own::Mat to_own(test::Mat mat) {
    return {mat.rows, mat.cols, mat.type, mat.data, mat.step};
}

std::vector<cv::gapi::own::Mat> to_own(std::vector<test::Mat> mats)
{
    std::vector<cv::gapi::own::Mat> own_mats(mats.size());
    for (int i = 0; i < mats.size(); i++) {
        own_mats[i] = to_own(mats[i]);
    }
    return own_mats;
}
} // anonymous namespace

struct ResizeComputation::ResizePriv
{
    cv::GCompiled m_cc;
};

static cv::GComputation buildResizeComputation(test::Mat outMat, int interp)
{
    cv::gapi::own::Size sz_out {outMat.cols, outMat.rows/3};
    cv::GMatP in;
    auto out = InferenceEngine::gapi::resizeP(in, sz_out, interp);
    return cv::GComputation(in, out);
}

ResizeComputation::ResizeComputation(test::Mat inMat, test::Mat outMat, int interp)
    : ComputationBase(new Priv{buildResizeComputation(outMat, interp)
                               ,{to_own(inMat)}
                               ,{to_own(outMat)}
                               })
    , m_resizePriv(new ResizePriv)
{}

void ResizeComputation::warmUp()
{
    m_resizePriv->m_cc = m_priv->m_c.compile(cv::gapi::own::descr_of(m_priv->m_v_in[0]).asPlanar(3),
                                             cv::compile_args(InferenceEngine::gapi::preproc::sipp::kernels()));
    apply();
}

void ResizeComputation::apply()
{
    m_resizePriv->m_cc(cv::gin(m_priv->m_v_in[0]), cv::gout(m_priv->m_v_out[0]));
}

static cv::GComputation buildNV12toRGBComputation()
{
    cv::GMat in_y;
    cv::GMat in_uv;
    cv::GMat out = InferenceEngine::gapi::NV12toRGBp(in_y, in_uv);
    return cv::GComputation(cv::GIn(in_y,in_uv), cv::GOut(out));
}

NV12toRGBComputation::NV12toRGBComputation(test::Mat inMat_y, test::Mat inMat_uv, test::Mat outMat)
    : ComputationBase(new Priv{buildNV12toRGBComputation()
                               ,to_own({inMat_y,inMat_uv})
                               ,{to_own(outMat)}
                               })
{}

struct MergeComputation::MergePriv
{
    cv::GCompiled m_cc;
};

static cv::GComputation buildMergeComputation()
{
    cv::GMatP in;
    auto out = InferenceEngine::gapi::merge3p(in);
    return cv::GComputation(in, out);
}

MergeComputation::MergeComputation(test::Mat inMat, test::Mat outMat)
    : ComputationBase(new Priv{buildMergeComputation()
                               ,{to_own(inMat)}
                               ,{to_own(outMat)}
                               })
    , m_mergePriv(new MergePriv)
{}

MergeComputation::~MergeComputation() = default;

void MergeComputation::warmUp()
{
    m_mergePriv->m_cc = m_priv->m_c.compile(cv::gapi::own::descr_of(m_priv->m_v_in[0]).asPlanar(3),
                                             cv::compile_args(InferenceEngine::gapi::preproc::sipp::kernels()));
    apply();
}

void MergeComputation::apply()
{
    m_mergePriv->m_cc(cv::gin(m_priv->m_v_in[0]), cv::gout(m_priv->m_v_out[0]));
}
