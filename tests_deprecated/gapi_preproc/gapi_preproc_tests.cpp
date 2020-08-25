// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_compound_blob.h>

#include <blob_factory.hpp>
#include <blob_transform.hpp>
#include <ie_preprocess.hpp>
#include <ie_preprocess_data.hpp>
#include <kmb_preproc.hpp>
#include <kmb_preproc_gapi.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

#include "gapi_test_computations.hpp"
#include "kmb_allocator.h"

#ifndef UNUSED
#define UNUSED(var) (void)var
#endif

// clang-format off
namespace {

void toPlanar(const cv::Mat& in, cv::Mat& out) {
    IE_ASSERT(out.depth() == in.depth());
    IE_ASSERT(out.channels() == 1);
    IE_ASSERT(in.channels() == 3);
    IE_ASSERT(out.cols == in.cols);
    IE_ASSERT(out.rows == 3 * in.rows);

    std::vector<cv::Mat> outs(3);
    for (int i = 0; i < 3; i++) {
        outs[i] = out(cv::Rect(0, i * in.rows, in.cols, in.rows));
    }
    cv::split(in, outs);
}

inline uchar saturate(int v) { return (v > 255 ? 255 : v < 0 ? 0 : v); }

inline uchar calcR(int y, int /*u*/, int v) {
    int r = y + (int)(1.402f * v);
    return saturate(r);
}

inline uchar calcG(int y, int u, int v) {
    int g = y - (int)(0.344f * u + 0.714 * v);
    return saturate(g);
}

inline uchar calcB(int y, int u, int /*v*/) {
    int b = y + (int)(1.772f * u);
    return saturate(b);
}

void own_NV12toBGR(const cv::Mat& inY, const cv::Mat& inUV, cv::Mat& out) {
    int i, j;

    uchar* y = inY.data;
    uchar* uv = inUV.data;
    uchar* bgr = out.data;

    uint yidx = 0;
    uint uvidx = 0;
    uint bgridx = 0;
    int yy, u, v;
    for (j = 0; j < inY.rows; j++) {
        y = inY.data + j * inY.step;
        yidx = 0;
        uv = inUV.data + (j / 2) * inUV.step;
        uvidx = 0;

        for (i = 0; i < inY.cols; i += 2) {
            u = uv[uvidx] - 128;
            v = uv[uvidx + 1] - 128;
            uvidx += 2;

            yy = y[yidx];
            yidx++;

            bgr[bgridx++] = calcB(yy, u, v);
            bgr[bgridx++] = calcG(yy, u, v);
            bgr[bgridx++] = calcR(yy, u, v);

            //----------------------------------------------
            yy = y[yidx];
            yidx++;

            bgr[bgridx++] = calcB(yy, u, v);
            bgr[bgridx++] = calcG(yy, u, v);
            bgr[bgridx++] = calcR(yy, u, v);
        }
    }
}

void own_NV12toRGB(const cv::Mat& inY, const cv::Mat& inUV, cv::Mat& out) {
    int i, j;

    uchar* y = inY.data;
    uchar* uv = inUV.data;
    uchar* rgb = out.data;

    uint yidx = 0;
    uint uvidx = 0;
    uint rgbidx = 0;
    int yy, u, v;
    for (j = 0; j < inY.rows; j++) {
        y = inY.data + j * inY.step;
        yidx = 0;
        uv = inUV.data + (j / 2) * inUV.step;
        uvidx = 0;
        for (i = 0; i < inY.cols; i += 2) {
            u = uv[uvidx] - 128;
            v = uv[uvidx + 1] - 128;
            uvidx += 2;

            yy = y[yidx];
            yidx++;

            rgb[rgbidx++] = calcR(yy, u, v);
            rgb[rgbidx++] = calcG(yy, u, v);
            rgb[rgbidx++] = calcB(yy, u, v);

            //----------------------------------------------
            yy = y[yidx];
            yidx++;

            rgb[rgbidx++] = calcR(yy, u, v);
            rgb[rgbidx++] = calcG(yy, u, v);
            rgb[rgbidx++] = calcB(yy, u, v);
        }
    }
}

class AllocHelper {
    const int defaultDeviceId = 0;
    vpu::KmbPlugin::KmbAllocator m_alloc{defaultDeviceId};
    std::vector<std::shared_ptr<void>> m_buffs;

public:
    void* alloc(size_t size) {
        void* ptr = m_alloc.alloc(size);
        m_buffs.push_back(std::shared_ptr<void>(ptr, [&](void* p) {
            m_alloc.free(p);
        }));
        return ptr;
    }
};

// FIXME: copy-paste from cropResize_tests.hpp
template <InferenceEngine::Precision::ePrecision PRC>
InferenceEngine::Blob::Ptr img2Blob(
    const std::vector<cv::Mat>& imgs, InferenceEngine::Layout layout, AllocHelper& allocator) {
    using data_t = typename InferenceEngine::PrecisionTrait<PRC>::value_type;
    using namespace InferenceEngine;

    if (imgs.empty()) {
        THROW_IE_EXCEPTION << "No images to create blob from";
    }

    // get image value in correct format
    static const auto img_value = [](const cv::Mat& img, size_t h, size_t w, size_t c) -> data_t {
        switch (img.type()) {
        case CV_8UC1:
            return img.at<uchar>(h, w);
        case CV_8UC2:
            return img.at<cv::Vec2b>(h, w)[c];
        case CV_8UC3:
            return img.at<cv::Vec3b>(h, w)[c];
        case CV_8UC4:
            return img.at<cv::Vec4b>(h, w)[c];
        case CV_32FC3:
            return img.at<cv::Vec3f>(h, w)[c];
        case CV_32FC4:
            return img.at<cv::Vec4f>(h, w)[c];
        default:
            THROW_IE_EXCEPTION << "Image type is not recognized";
        }
    };

    size_t channels = imgs[0].channels();
    size_t height = imgs[0].size().height;
    size_t width = imgs[0].size().width;

    SizeVector dims = {imgs.size(), channels, height, width};
    auto buf = reinterpret_cast<data_t*>(allocator.alloc(width * height * channels));
    Blob::Ptr resultBlob = make_shared_blob<data_t>(TensorDesc(PRC, dims, layout), buf);

    data_t* blobData = resultBlob->buffer().as<data_t*>();

    for (size_t i = 0; i < imgs.size(); ++i) {
        auto& img = imgs[i];
        auto batch_offset = i * channels * height * width;

        switch (layout) {
        case Layout::NCHW: {
            for (size_t c = 0; c < channels; c++) {
                for (size_t h = 0; h < height; h++) {
                    for (size_t w = 0; w < width; w++) {
                        blobData[batch_offset + c * width * height + h * width + w] = img_value(img, h, w, c);
                    }
                }
            }
        } break;
        case Layout::NHWC: {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    for (size_t c = 0; c < channels; c++) {
                        blobData[batch_offset + h * width * channels + w * channels + c] = img_value(img, h, w, c);
                    }
                }
            }
        } break;
        default:
            THROW_IE_EXCEPTION << "Inconsistent input layout for image processing: " << layout;
        }
    }
    return resultBlob;
}

template <InferenceEngine::Precision::ePrecision PRC>
InferenceEngine::Blob::Ptr img2Blob(cv::Mat& img, InferenceEngine::Layout layout, AllocHelper& allocator) {
    return img2Blob<PRC>(std::vector<cv::Mat>({img}), layout, allocator);
}

template <InferenceEngine::Precision::ePrecision PRC>
void Blob2Img(const InferenceEngine::Blob::Ptr& blobP, cv::Mat& img, InferenceEngine::Layout layout) {
    using namespace InferenceEngine;
    using data_t = typename PrecisionTrait<PRC>::value_type;

    const size_t channels = img.channels();
    const size_t height = img.size().height;
    const size_t width = img.size().width;

    CV_Assert(cv::DataType<data_t>::depth == img.depth());

    data_t* blobData = blobP->buffer().as<data_t*>();

    switch (layout) {
    case Layout::NCHW: {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    img.ptr<data_t>(h, w)[c] = blobData[c * width * height + h * width + w];
                }
            }
        }
    } break;
    case Layout::NHWC: {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                for (size_t c = 0; c < channels; c++) {
                    img.ptr<data_t>(h, w)[c] = blobData[h * width * channels + w * channels + c];
                }
            }
        }
    } break;
    default:
        THROW_IE_EXCEPTION << "Inconsistent input layout for image processing: " << layout;
    }
}

test::Mat to_test(cv::Mat& mat) { return {mat.rows, mat.cols, mat.type(), mat.data, mat.step}; }

InferenceEngine::ROI to_ie(cv::Rect roi) {
    InferenceEngine::ROI ie_roi;
    ie_roi.posX = roi.x;
    ie_roi.posY = roi.y;
    ie_roi.sizeX = roi.width;
    ie_roi.sizeY = roi.height;
    return ie_roi;
}

cv::Rect getRandomRoi(cv::Size size) {
    cv::Rect rect;
    auto getRand = []() {
        return ((double)std::rand() / (double)RAND_MAX);
    };

    rect.x = (size.width - 64) * getRand();
    rect.y = (size.height - 64) * getRand();
    rect.width = (size.width / 4) * getRand();
    rect.height = (size.height / 4) * getRand();

    if (rect.width < 64) rect.width = 64;
    if (rect.height < 64) rect.height = 64;
    if (rect.width % 2 == 1) rect.width += 1;
    if (rect.height % 2 == 1) rect.height += 1;

    rect.x = (size.width - rect.width) * getRand();
    rect.y = (size.height - rect.height) * getRand();

    return rect;
}

cv::Rect getFullImageRoi(cv::Size size) {
    cv::Rect rect;
    rect.x = 0;
    rect.y = 0;
    rect.width = size.width;
    rect.height = size.height;
    return rect;
}

std::pair<InferenceEngine::Blob::Ptr, InferenceEngine::Blob::Ptr> img2NV12Blobs(const cv::Mat& img, AllocHelper& allocator)
{
    IE_ASSERT(img.channels() == 1);
    using namespace InferenceEngine;
    constexpr auto prec = Precision::U8;
    constexpr auto layout = Layout::NHWC;
    using data_t = typename InferenceEngine::PrecisionTrait<prec>::value_type;
    size_t yW = img.cols;
    size_t yH = img.rows/3*2;
    size_t uvW = yW/2;
    size_t uvH = yH/2;

    SizeVector  yDims{1, 1,  yH,  yW};
    SizeVector uvDims{1, 2, uvH, uvW};

    auto buf = reinterpret_cast<data_t*>(allocator.alloc(img.rows*img.cols));
    Blob::Ptr  yBlob = make_shared_blob<data_t>(TensorDesc(prec,  yDims, layout), buf);
    Blob::Ptr uvBlob = make_shared_blob<data_t>(TensorDesc(prec, uvDims, layout), buf + yW*yH);

    std::copy_n(img.data, img.rows*img.cols, buf);

    return std::make_pair(yBlob, uvBlob);
}
}  // anonymous namespace

struct NV12toRGBpTestGAPI : public testing::TestWithParam<cv::Size> {};
TEST_P(NV12toRGBpTestGAPI, AccuracyTest) {
    cv::Size sz_y = GetParam();
    cv::Size sz_uv = cv::Size(sz_y.width / 2, sz_y.height / 2);
    cv::Size sz_p = cv::Size(sz_y.width, sz_y.height * 3);

    cv::Mat in_mat_y(sz_y, CV_8UC1);
    cv::Mat in_mat_uv(sz_uv, CV_8UC2);
    cv::randn(in_mat_y, cv::Scalar::all(127), cv::Scalar::all(40.f));
    cv::randn(in_mat_uv, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::Mat out_mat_gapi(cv::Mat::zeros(sz_p, CV_8UC1));
    cv::Mat out_mat_ocv(cv::Mat::zeros(sz_p, CV_8UC1));

    // G-API code //////////////////////////////////////////////////////////////
    NV12toRGBComputation sc(to_test(in_mat_y), to_test(in_mat_uv), to_test(out_mat_gapi));
    sc.warmUp();

#if PERF_TEST
    // iterate testing, and print performance
    test_ms(
        [&]() {
            sc.apply();
        },
        400, "NV12toRGB GAPI %s %dx%d", typeToString(CV_8UC3).c_str(), sz.width, sz.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Mat out_mat_ocv_interleaved(cv::Mat::zeros(sz_y, CV_8UC3));
        own_NV12toRGB(in_mat_y, in_mat_uv, out_mat_ocv_interleaved);
        // cv::cvtColorTwoPlane(in_mat_y, in_mat_uv, out_mat_ocv_interleaved, cv::COLOR_YUV2RGB_NV12);
        toPlanar(out_mat_ocv_interleaved, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    { EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi)); }
}

using testing::Values;

INSTANTIATE_TEST_CASE_P(NV12toRGBTestSIPP, NV12toRGBpTestGAPI,
    Values(cv::Size(224, 224), cv::Size(1280, 720), cv::Size(1280, 960), cv::Size(960, 720), cv::Size(640, 480),
        cv::Size(300, 300), cv::Size(320, 200)));

struct ResizePTestGAPI : public testing::TestWithParam<std::pair<cv::Size, cv::Size>> {};
TEST_P(ResizePTestGAPI, AccuracyTest) {
    constexpr int planeNum = 3;
    cv::Size sz_in, sz_out;
    std::tie(sz_in, sz_out) = GetParam();

    auto interp = cv::INTER_LINEAR;

    cv::Size sz_in_p(sz_in.width, sz_in.height * 3);
    cv::Size sz_out_p(sz_out.width, sz_out.height * 3);

    cv::Mat in_mat(sz_in_p, CV_8UC1);
    cv::randn(in_mat, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::Mat out_mat_gapi(cv::Mat::zeros(sz_out_p, CV_8UC1));
    cv::Mat out_mat_ocv(cv::Mat::zeros(sz_out_p, CV_8UC1));

    // G-API code //////////////////////////////////////////////////////////////
    ResizeComputation sc(to_test(in_mat), to_test(out_mat_gapi), interp);
    sc.warmUp();

    // FIXME: perf compilation is likely needs to be fixed

#if PERF_TEST
    // iterate testing, and print performance
    test_ms(
        [&]() {
            sc.apply();
        },
        400, "NV12toRGB GAPI %s %dx%d", typeToString(CV_8UC3).c_str(), sz.width, sz.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        for (int i = 0; i < planeNum; i++) {
            const cv::Mat in_mat_roi = in_mat(cv::Rect(0, i * sz_in.height, sz_in.width, sz_in.height));
            cv::Mat out_mat_roi = out_mat_ocv(cv::Rect(0, i * sz_out.height, sz_out.width, sz_out.height));
            cv::resize(in_mat_roi, out_mat_roi, sz_out, 0, 0, interp);
        }
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        cv::Mat absDiff;
        cv::absdiff(out_mat_gapi, out_mat_ocv, absDiff);
        EXPECT_EQ(0, cv::countNonZero(absDiff > 1));
    }
}

#define TEST_SIZES_PREPROC                                        \
    std::make_pair(cv::Size(1920, 1080), cv::Size(1024, 1024)),   \
        std::make_pair(cv::Size(1920, 1080), cv::Size(224, 224)), \
        std::make_pair(cv::Size(1280, 720), cv::Size(544, 320)),  \
        std::make_pair(cv::Size(640, 480), cv::Size(896, 512)),   \
        std::make_pair(cv::Size(200, 400), cv::Size(128, 384)),   \
        std::make_pair(cv::Size(256, 256), cv::Size(256, 256)), std::make_pair(cv::Size(96, 256), cv::Size(128, 384))

INSTANTIATE_TEST_CASE_P(ResizePTestSIPP, ResizePTestGAPI, Values(TEST_SIZES_PREPROC));


struct Merge3PTestGAPI: public testing::TestWithParam<cv::Size> {};
TEST_P(Merge3PTestGAPI, AccuracyTest)
{
    auto sz = GetParam();
    cv::Size sz_in_p (sz.width, sz.height *3);

    auto allocMat = [](cv::Size sz, int type, AllocHelper& allocator) {
        return cv::Mat(sz, type, allocator.alloc(sz.width*sz.height*CV_ELEM_SIZE(type)));
    };

    AllocHelper allocator;

    cv::Mat in_mat = allocMat(sz_in_p, CV_8UC1, allocator);
    cv::randn(in_mat, cv::Scalar::all(127), cv::Scalar::all(40.f));

    cv::Mat out_mat_gapi = allocMat(sz, CV_8UC3, allocator);
    cv::Mat out_mat_ocv  = allocMat(sz, CV_8UC3, allocator);

    // G-API code //////////////////////////////////////////////////////////////
    MergeComputation mc(to_test(in_mat), to_test(out_mat_gapi));
    mc.warmUp();

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        constexpr const int planeNum = 3;
        std::array<cv::Mat, planeNum> ins;
        for (int i = 0; i < planeNum; i++) {
            ins[i] = in_mat(cv::Rect(0, i*sz.height, sz.width, sz.height));
        }
        cv::merge(ins, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        cv::Mat absDiff;
        cv::absdiff(out_mat_gapi, out_mat_ocv, absDiff);
        EXPECT_EQ(0, cv::countNonZero(absDiff > 1));
    }

    std::cout << in_mat << std::endl << std::endl;
    std::cout << out_mat_ocv << std::endl;
}

INSTANTIATE_TEST_CASE_P(Merge3PTestSIPP, Merge3PTestGAPI,
                        Values(cv::Size(32, 8)));

using namespace testing;
std::ostream& operator << (std::ostream& os, InferenceEngine::KmbPreproc::Path path) {
    using InferenceEngine::KmbPreproc::Path;
    switch (path) {
    case Path::SIPP: os << "SIPP"; return os;
    case Path::M2I : os << "M2I" ; return os;
    default: IE_ASSERT(false);
    }
}

struct KmbPreprocEngineTest : public TestWithParam<std::tuple<
    std::pair<cv::Size, cv::Size>,
    InferenceEngine::ColorFormat,
    InferenceEngine::KmbPreproc::Path>> {};
TEST_P(KmbPreprocEngineTest, TestNV12Resize) {
    using namespace InferenceEngine;

    constexpr auto prec = Precision::U8;
    ResizeAlgorithm interp = RESIZE_BILINEAR;
    Layout in_layout = Layout::NCHW;
    Layout out_layout = in_layout;
    ColorFormat in_fmt = ColorFormat::NV12;
    ColorFormat out_fmt;
    std::pair<cv::Size,cv::Size> sizes;
    KmbPreproc::Path preprocPath;
    std::tie(sizes, out_fmt, preprocPath) = GetParam();
    cv::Size y_size, out_size;
    std::tie(y_size, out_size) = sizes;
    cv::Size uv_size {y_size.width / 2, y_size.height / 2};

    cv::Mat in_mat(cv::Size{y_size.width, y_size.height*3/2}, CV_8UC1);
    cv::Mat y_mat(y_size, CV_8UC1, in_mat.data);
    cv::Mat uv_mat(uv_size, CV_8UC2, in_mat.ptr<uchar>(y_mat.rows, 0));
    cv::Mat out_mat(out_size, CV_8UC3);
    cv::randu(y_mat, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(uv_mat, cv::Scalar::all(0), cv::Scalar::all(255));

    AllocHelper allocator;
    auto nv12blobs = (preprocPath == KmbPreproc::Path::SIPP)
                     ? std::make_pair(img2Blob<prec>( y_mat, Layout::NHWC, allocator),
                                      img2Blob<prec>(uv_mat, Layout::NHWC, allocator))
                     : img2NV12Blobs(in_mat, allocator);
    auto  y_blob = nv12blobs.first;
    auto uv_blob = nv12blobs.second;

    auto out_blob = img2Blob<prec>(out_mat, out_layout, allocator);

    unsigned int shaveFirst = 0;
    unsigned int shaveLast = 3;
    unsigned int lpi = 8;
    KmbPreproc::PreprocEngine pe(shaveFirst, shaveLast, lpi, preprocPath);

    for (int i = 0; i < 10; i++) {
        auto y_roi = (preprocPath == KmbPreproc::Path::SIPP)
                     ? getRandomRoi(y_size)
                     : getFullImageRoi(y_size);

        cv::Rect uv_roi {y_roi.x / 2, y_roi.y / 2, y_roi.width / 2, y_roi.height / 2};

        auto y_roi_blob = make_shared_blob(y_blob, to_ie(y_roi));
        auto uv_roi_blob = make_shared_blob(uv_blob, to_ie(uv_roi));

        auto in_blob = make_shared_blob<NV12Blob>(y_roi_blob, uv_roi_blob);

        const int deviceId = 0;
        pe.preproc(in_blob, out_blob, interp, in_fmt, out_fmt, deviceId);

        Blob2Img<prec>(out_blob, out_mat, out_layout);

        cv::Mat rgb_mat(cv::Size {y_roi.width, y_roi.height}, CV_8UC3);
        cv::Mat ocv_out_mat(out_size, CV_8UC3);

        if (out_fmt == ColorFormat::RGB) {
            own_NV12toRGB(y_mat(y_roi), uv_mat(uv_roi), rgb_mat);
        } else {
            own_NV12toBGR(y_mat(y_roi), uv_mat(uv_roi), rgb_mat);
        }
        cv::resize(rgb_mat, ocv_out_mat, out_size, 0, 0, cv::INTER_LINEAR);

        EXPECT_TRUE(cv::norm(ocv_out_mat, out_mat, cv::NORM_INF) <= 2);
    }
}

INSTANTIATE_TEST_CASE_P(Preproc, KmbPreprocEngineTest,
                        Combine(Values(std::make_pair(cv::Size(1920, 1080), cv::Size(416, 416))),
                                Values(InferenceEngine::ColorFormat::BGR,
                                       InferenceEngine::ColorFormat::RGB),
                                Values(InferenceEngine::KmbPreproc::Path::SIPP)));

// FIXME: hsdes ticket https://hsdes.intel.com/appstore/article/#/1508160288
INSTANTIATE_TEST_CASE_P(DISABLED_Preproc, KmbPreprocEngineTest,
                        Combine(Values(std::make_pair(cv::Size(1920, 1080), cv::Size(224, 224))),
                                Values(InferenceEngine::ColorFormat::BGR,
                                       InferenceEngine::ColorFormat::RGB),
                                Values(InferenceEngine::KmbPreproc::Path::SIPP)));

// FIXME: doesn't converge with opencv, need to figure out tollerance
INSTANTIATE_TEST_CASE_P(DISABLED_PreprocM2I, KmbPreprocEngineTest,
                        Combine(Values(std::make_pair(cv::Size(1920, 1080), cv::Size(416, 416))),
                                Values(InferenceEngine::ColorFormat::BGR),
                                Values(InferenceEngine::KmbPreproc::Path::M2I)));

struct KmbPreprocPoolTest: public TestWithParam<std::tuple<
    std::tuple<cv::Size, cv::Size, cv::Size>,
    InferenceEngine::Layout,
    InferenceEngine::ColorFormat,
    InferenceEngine::KmbPreproc::Path>> {};
TEST_P(KmbPreprocPoolTest, TestNV12Resize)
{
    using namespace InferenceEngine;

    Layout out_layout;
    ColorFormat out_fmt;
    KmbPreproc::Path detectionPreprocPath;
    std::tuple<cv::Size,cv::Size,cv::Size> sizes;
    std::tie(sizes, out_layout, out_fmt, detectionPreprocPath) = GetParam();
    cv::Size y_size, detect_size, classify_size;
    std::tie(y_size, detect_size, classify_size) = sizes;

    constexpr int numThreads = 8;

    enum class NNKind : int {
        Classify = 0,
        Detect
    };

    struct TestContext {
        Layout out_layout;
        NNKind nnKind;
        KmbPreproc::Path preprocPath;

        cv::Mat in_mat;
        cv::Mat y_mat;
        cv::Mat uv_mat;
        cv::Mat out_mat;

        Blob::Ptr y_blob;
        Blob::Ptr uv_blob;
        Blob::Ptr out_blob;

        std::string inName;
        InputsDataMap inputInfos;
        std::map<std::string, PreProcessDataPtr> preprocDatas;
        BlobMap netInputs;

        void init(cv::Size y_size, cv::Size out_size, Layout output_layout, AllocHelper& allocator, NNKind kind,
                  KmbPreproc::Path path = KmbPreproc::Path::SIPP) {
            IE_ASSERT(!(path == KmbPreproc::Path::M2I && kind == NNKind::Classify));
            nnKind = kind;
            preprocPath = path;
            out_layout = output_layout;
            constexpr auto prec = Precision::U8;
            cv::Size uv_size {y_size.width / 2, y_size.height / 2};

            in_mat = cv::Mat(cv::Size{y_size.width, y_size.height*3/2}, CV_8UC1);
            y_mat = cv::Mat(y_size, CV_8UC1, in_mat.data);
            uv_mat = cv::Mat(uv_size, CV_8UC2, in_mat.ptr<uchar>(y_mat.rows, 0));
            out_mat = cv::Mat(out_size, CV_8UC3);
            cv::randu(y_mat, cv::Scalar::all(0), cv::Scalar::all(255));
            cv::randu(uv_mat, cv::Scalar::all(128), cv::Scalar::all(128));

            auto nv12blobs = (preprocPath == KmbPreproc::Path::SIPP)
                             ? std::make_pair(img2Blob<prec>( y_mat, Layout::NHWC, allocator),
                                              img2Blob<prec>(uv_mat, Layout::NHWC, allocator))
                             : img2NV12Blobs(in_mat, allocator);
            y_blob  = nv12blobs.first;
            uv_blob = nv12blobs.second;
            out_blob = img2Blob<prec>(out_mat, out_layout, allocator);

            inName = "input0";
            inputInfos[inName] = std::make_shared<InputInfo>();
            SizeVector dims {static_cast<size_t>(y_size.width), static_cast<size_t>(y_size.height), 1, 1};
            inputInfos[inName]->setInputData(std::make_shared<InferenceEngine::Data>(inName,
                InferenceEngine::TensorDesc { prec, dims, Layout::NHWC }));
            inputInfos[inName]->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
            inputInfos[inName]->getPreProcess().setColorFormat(NV12);
            inputInfos[inName]->setLayout(Layout::NHWC);

            preprocDatas[inName] = CreatePreprocDataHelper();
        }
    };

    AllocHelper allocator;

    TestContext detectContexts[numThreads];
    for (int i = 0; i < numThreads; i++) {
        detectContexts[i].init(y_size, detect_size, out_layout, allocator, NNKind::Detect, detectionPreprocPath);
    }

    TestContext classifyContexts[numThreads];
    for (int i = 0; i < numThreads; i++) {
        classifyContexts[i].init(y_size, classify_size, out_layout, allocator, NNKind::Classify);
    }

    auto threadFunc = [y_size, out_layout, out_fmt](TestContext& ctx, cv::Size out_size) {
        UNUSED(out_size);
        for (int i = 0; i < 10; i++) {
            auto y_roi = ctx.nnKind == (NNKind::Classify) ? getRandomRoi(y_size) : getFullImageRoi(y_size);

            cv::Rect uv_roi {y_roi.x / 2, y_roi.y / 2, y_roi.width / 2, y_roi.height / 2};

            auto y_roi_blob = make_shared_blob(ctx.y_blob, to_ie(y_roi));
            auto uv_roi_blob = make_shared_blob(ctx.uv_blob, to_ie(uv_roi));

            auto in_blob = make_shared_blob<NV12Blob>(y_roi_blob, uv_roi_blob);

            ctx.preprocDatas[ctx.inName]->setRoiBlob(in_blob);
            ctx.netInputs[ctx.inName] = ctx.out_blob;

            unsigned int nShaves = 4;
            unsigned int lpi = 8;
            const std::string graphId = "1";
            const int deviceId = 0;
            KmbPreproc::execDataPreprocessing(
                ctx.netInputs, ctx.preprocDatas, ctx.inputInfos, out_fmt, nShaves, lpi, graphId, deviceId, ctx.preprocPath);
        }

#if 0
        Blob2Img<prec>(ctx.out_blob, ctx.out_mat, out_layout);

        cv::Mat rgb_mat(cv::Size{y_roi.width, y_roi.height}, CV_8UC3);
        cv::Mat ocv_out_mat(out_size, CV_8UC3);

        own_NV12toRGB(ctx.y_mat(y_roi), ctx.uv_mat(uv_roi), rgb_mat);
        cv::resize(rgb_mat, ocv_out_mat, out_size, 0, 0, cv::INTER_LINEAR);

        cv::Mat absDiff;
        cv::absdiff(ocv_out_mat, ctx.out_mat, absDiff);
        EXPECT_EQ(cv::countNonZero(absDiff > 1), 0);
#endif
    };

    std::thread detectThreads[numThreads];
    std::thread classifyThreads[numThreads];

    for (int t = 0; t < numThreads; t++) {
        detectThreads[t] = std::thread([t, &threadFunc, &detectContexts, detect_size]() {
            threadFunc(detectContexts[t], detect_size);
        });
        classifyThreads[t] = std::thread([t, &threadFunc, &classifyContexts, classify_size]() {
            threadFunc(classifyContexts[t], classify_size);
        });
    }

    for (int i = 0; i < numThreads; i++) {
        detectThreads[i].join();
        classifyThreads[i].join();
    }
}

INSTANTIATE_TEST_CASE_P(Preproc, KmbPreprocPoolTest,
                        Combine(Values(std::make_tuple(cv::Size(1920, 1080),
                                                       cv::Size(416, 416),
                                                       cv::Size(224, 224))),
                                Values(InferenceEngine::Layout::NCHW,
                                       InferenceEngine::Layout::NHWC),
                                Values(InferenceEngine::ColorFormat::BGR,
                                       InferenceEngine::ColorFormat::RGB),
                                Values(InferenceEngine::KmbPreproc::Path::SIPP)));

INSTANTIATE_TEST_CASE_P(DISABLED_PreprocM2I, KmbPreprocPoolTest,
                        Combine(Values(std::make_tuple(cv::Size(1920, 1080),
                                                       cv::Size(416, 416),
                                                       cv::Size(224, 224))),
                                Values(InferenceEngine::Layout::NCHW,
                                       InferenceEngine::Layout::NHWC),
                                Values(InferenceEngine::ColorFormat::BGR),
                                Values(InferenceEngine::KmbPreproc::Path::M2I)));

// clang-format on
