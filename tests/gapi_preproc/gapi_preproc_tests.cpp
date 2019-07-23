// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi.hpp>

#include "gapi_test_computations.hpp"

namespace {

void toPlanar(const cv::Mat& in, cv::Mat& out)
{
    GAPI_Assert(out.depth() == in.depth());
    GAPI_Assert(out.channels() == 1);
    GAPI_Assert(in.channels() == 3);
    GAPI_Assert(out.cols == in.cols);
    GAPI_Assert(out.rows == 3*in.rows);

    std::vector<cv::Mat> outs(3);
    for (int i = 0; i < 3; i++) {
        outs[i] = out(cv::Rect(0, i*in.rows, in.cols, in.rows));
    }
    cv::split(in, outs);
}

void own_NV12toBGR(const cv::Mat& inY, const cv::Mat& inUV, cv::Mat& out)
{
    int i ,j;

    uchar* y  = inY.data;
    uchar* uv = inUV.data;
    uchar* bgr = out.data;

    uint yidx = 0;
    uint uvidx = 0;
    uint bgridx = 0;
    int yy,u,v,r,g,b;
    for(j = 0; j < inY.rows; j++)
    {
        y = inY.data + j*inY.step;
        yidx = 0;
        uv = inUV.data + (j/2)*inUV.step;
        uvidx = 0;

        for(i = 0; i < inY.cols; i+=2 )
        {
            yy = y[yidx];
            yidx++;

            u = uv[uvidx] - 128;
            v = uv[uvidx+1] - 128;
            uvidx += 2;
            b =  yy + (int)(1.772f*u);

            bgr[bgridx++] = (uchar) (b > 255 ? 255 : b < 0 ? 0 : b);
            g =  yy - (int)(0.344f*u + 0.714*v);


            bgr[bgridx++] = (uchar) (g > 255 ? 255 : g < 0 ? 0 : g);
            r =  yy + (int)(1.402f*v);

            bgr[bgridx++] = (uchar) (r > 255 ? 255 : r < 0 ? 0 : r);
            //----------------------------------------------
            yy = y[yidx];
            yidx++;
            b =  yy + (int)(1.772f*u);

            bgr[bgridx++] = (uchar) (b > 255 ? 255 : b < 0 ? 0 : b);
            g =  yy - (int)(0.344f*u + 0.714*v);

            bgr[bgridx++] = (uchar) (g > 255 ? 255 : g < 0 ? 0 : g);
            r =  yy + (int)(1.402f*v);

            bgr[bgridx++] = (uchar) (r > 255 ? 255 : r < 0 ? 0 : r);
        }

    }
}

void own_NV12toRGB(const cv::Mat& inY, const cv::Mat& inUV, cv::Mat& out)
{
    int i ,j;

    uchar* y  = inY.data;
    uchar* uv = inUV.data;
    uchar* rgb = out.data;

    uint yidx = 0;
    uint uvidx = 0;
    uint rgbidx = 0;
    int yy,u,v,r,g,b;
    for(j = 0; j < inY.rows; j++)
    {

        y = inY.data + j*inY.step;
        yidx = 0;
        uv = inUV.data + (j/2)*inUV.step;
        uvidx = 0;
        for(i = 0; i < inY.cols; i+=2 )
        {
            yy = y[yidx];
            yidx++;

            u = uv[uvidx] - 128;
            v = uv[uvidx+1] - 128;
            uvidx += 2;
            r =  yy + (int)(1.772f*u);

            rgb[rgbidx++] = (uchar) (r > 255 ? 255 : r < 0 ? 0 : r);
            g =  yy - (int)(0.344f*u + 0.714*v);


            rgb[rgbidx++] = (uchar) (g > 255 ? 255 : g < 0 ? 0 : g);
            b =  yy + (int)(1.402f*v);

            rgb[rgbidx++] = (uchar) (b > 255 ? 255 : b < 0 ? 0 : b);
            //----------------------------------------------
            yy = y[yidx];
            yidx++;
            r =  yy + (int)(1.772f*u);

            rgb[rgbidx++] = (uchar) (r > 255 ? 255 : r < 0 ? 0 : r);
            g =  yy - (int)(0.344f*u + 0.714*v);

            rgb[rgbidx++] = (uchar) (g > 255 ? 255 : g < 0 ? 0 : g);
            b =  yy + (int)(1.402f*v);

            rgb[rgbidx++] = (uchar) (b > 255 ? 255 : b < 0 ? 0 : b);
        }

    }
}

test::Mat to_test(cv::Mat& mat) { return {mat.rows, mat.cols, mat.type(), mat.data, mat.step}; }

} // anonymous namespace

struct NV12toRGBpTestGAPI: public testing::TestWithParam<cv::Size> {};
TEST_P(NV12toRGBpTestGAPI, AccuracyTest)
{
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
    test_ms([&](){ sc.apply(); },
        400, "NV12toRGB GAPI %s %dx%d", typeToString(CV_8UC3).c_str(), sz.width, sz.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        cv::Mat out_mat_ocv_interleaved(cv::Mat::zeros(sz_y, CV_8UC3));
        own_NV12toRGB(in_mat_y, in_mat_uv, out_mat_ocv_interleaved);
        //cv::cvtColorTwoPlane(in_mat_y, in_mat_uv, out_mat_ocv_interleaved, cv::COLOR_YUV2RGB_NV12);
        toPlanar(out_mat_ocv_interleaved, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cv::countNonZero(out_mat_ocv != out_mat_gapi));
    }

    cv::imwrite("out_mat_ocv.jpg", out_mat_ocv);
    cv::imwrite("out_mat_gapi.jpg", out_mat_gapi);
}

using testing::Values;

INSTANTIATE_TEST_CASE_P(NV12toRGBTestSIPP, NV12toRGBpTestGAPI,
                        Values(cv::Size(224, 224)/*,
                               cv::Size(1280,  720),
                               cv::Size(1280,  960),
                               cv::Size( 960,  720),
                               cv::Size( 640,  480),
                               cv::Size( 300,  300),
                               cv::Size( 320,  200)*/));

struct ResizePTestGAPI: public testing::TestWithParam<std::pair<cv::Size, cv::Size>> {};
TEST_P(ResizePTestGAPI, AccuracyTest)
{
    constexpr int planeNum = 3;
    cv::Size sz_in, sz_out;
    std::tie(sz_in, sz_out) = GetParam();

    auto interp = cv::INTER_LINEAR;

    cv::Size sz_in_p (sz_in.width,  sz_in.height *3);
    cv::Size sz_out_p(sz_out.width, sz_out.height*3);

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
    test_ms([&](){ sc.apply(); },
        400, "NV12toRGB GAPI %s %dx%d", typeToString(CV_8UC3).c_str(), sz.width, sz.height);
#endif

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        for (int i = 0; i < planeNum; i++) {
            const cv::Mat in_mat_roi = in_mat(cv::Rect(0, i*sz_in.height,  sz_in.width,  sz_in.height));
            cv::Mat out_mat_roi = out_mat_ocv(cv::Rect(0, i*sz_out.height, sz_out.width, sz_out.height));
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

#define TEST_SIZES_PREPROC                                      \
    std::make_pair(cv::Size(1920, 1080), cv::Size(1024, 1024)), \
    std::make_pair(cv::Size(1920, 1080), cv::Size( 224,  224)), \
    std::make_pair(cv::Size(1280,  720), cv::Size( 544,  320)), \
    std::make_pair(cv::Size( 640,  480), cv::Size( 896,  512)), \
    std::make_pair(cv::Size( 200,  400), cv::Size( 128,  384)), \
    std::make_pair(cv::Size( 256,  256), cv::Size( 256,  256)), \
    std::make_pair(cv::Size(  96,  256), cv::Size( 128,  384))

INSTANTIATE_TEST_CASE_P(ResizePTestSIPP, ResizePTestGAPI,
                        Values(TEST_SIZES_PREPROC));
