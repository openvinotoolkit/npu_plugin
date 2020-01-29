// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#ifndef GAPI_TEST_COMPUTATIONS_HPP
#define GAPI_TEST_COMPUTATIONS_HPP

#include <ie_api.h>

#include <memory>
#include <vector>

#if defined(_WIN32)
#ifdef IMPLEMENT_GAPI_COMPUTATION_API
#define GAPI_COMPUTATION_VISIBILITY __declspec(dllexport)
#else
#define GAPI_COMPUTATION_VISIBILITY __declspec(dllimport)
#endif
#else
#ifdef IMPLEMENT_GAPI_COMPUTATION_API
#define GAPI_COMPUTATION_VISIBILITY __attribute__((visibility("default")))
#else
#define GAPI_COMPUTATION_VISIBILITY
#endif
#endif

namespace test {
struct Mat {
    int rows;
    int cols;
    int type;
    void* data;
    size_t step;
};
struct Rect {
    int x;
    int y;
    int width;
    int height;
    bool empty() { return width == 0 && height == 0; };
};
}  // namespace test

class GAPI_COMPUTATION_VISIBILITY ComputationBase {
protected:
    struct Priv;
    std::shared_ptr<Priv> m_priv;

public:
    ComputationBase(Priv* priv);
    void warmUp();
    void apply();
};

class GAPI_COMPUTATION_VISIBILITY ResizeComputation : public ComputationBase {
    struct ResizePriv;
    std::shared_ptr<ResizePriv> m_resizePriv;

public:
    ResizeComputation(test::Mat inMat, test::Mat outMat, int interp);
    void warmUp();
    void apply();
};

class GAPI_COMPUTATION_VISIBILITY NV12toRGBComputation : public ComputationBase {
public:
    NV12toRGBComputation(test::Mat inMat_y, test::Mat inMat_uv, test::Mat outMat);
};

class GAPI_COMPUTATION_VISIBILITY MergeComputation : public ComputationBase
{
    struct MergePriv;
    std::unique_ptr<MergePriv> m_mergePriv;
public:
    MergeComputation(test::Mat inMat, test::Mat outMat);
    ~MergeComputation();
    void warmUp();
    void apply();
};

#endif // GAPI_TEST_COMPUTATIONS_HPP
// clang-format on
