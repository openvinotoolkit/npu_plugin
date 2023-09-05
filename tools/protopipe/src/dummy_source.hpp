//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <memory>
#include <thread>

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/streaming/source.hpp>  // cv::gapi::wip::IStreamSource

#include "timer.hpp"
#include "utils.hpp"

class DummySource final : public cv::gapi::wip::IStreamSource {
public:
    template <typename DurationT>
    DummySource(const DurationT latency, const cv::Mat& mat);

    bool pull(cv::gapi::wip::Data& data) override;
    cv::GMetaArg descr_of() const override;

    using ts_t = std::chrono::microseconds;
    void setDropFrames(const bool drop_frames);

private:
    int64_t m_latency;
    bool m_drop_frames;
    IWaitable::Ptr m_timer;

    cv::Mat m_mat;
    int64_t m_next_tick_ts = -1;
    int64_t m_curr_seq_id = 0;
};

template <typename DurationT>
DummySource::DummySource(const DurationT latency, const cv::Mat& mat)
        : m_latency(std::chrono::duration_cast<ts_t>(latency).count()),
          m_drop_frames(false),
          m_timer(IWaitable::create()),
          m_mat(mat) {
}
