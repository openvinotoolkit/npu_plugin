//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dummy_source.hpp"

bool DummySource::pull(cv::gapi::wip::Data& data) {
    using namespace std::chrono;
    using namespace cv::gapi::streaming;

    // NB: Wait m_latency before return the first frame.
    if (m_next_tick_ts == -1) {
        m_next_tick_ts = utils::timestamp<ts_t>() + m_latency;
    }

    int64_t curr_ts = utils::timestamp<ts_t>();
    if (curr_ts < m_next_tick_ts) {
        /*
         *            curr_ts
         *               |
         *    ------|----*-----|------->
         *                     ^
         *               m_next_tick_ts
         *
         *
         * NB: New frame will be produced at the m_next_tick_ts point.
         */
        m_timer->wait(ts_t{m_next_tick_ts - curr_ts});
    } else if (m_latency != 0) {
        /*
         *                                       curr_ts
         *                         +1         +2    |
         *    |----------|----------|----------|----*-----|------->
         *               ^                     ^
         *         m_next_tick_ts ------------->
         *
         */

        // NB: Count how many frames have been produced since last pull (m_next_tick_ts).
        int64_t num_frames = static_cast<int64_t>((curr_ts - m_next_tick_ts) / m_latency);
        // NB: Shift m_next_tick_ts to the nearest tick before curr_ts.
        m_next_tick_ts += num_frames * m_latency;
        // NB: if drop_frames is enabled, update current seq_id and wait for the next tick, otherwise
        // return last written frame (+2 at the picture above) immediately.
        if (m_drop_frames) {
            // NB: Shift tick to the next frame.
            m_next_tick_ts += m_latency;
            // NB: Wait for the next frame.
            m_timer->wait(ts_t{m_next_tick_ts - curr_ts});
            // NB: Drop already produced frames + update seq_id for the current.
            m_curr_seq_id += num_frames + 1;
        }
    }
    // NB: Just increase reference counter not to release mat memory
    // after assigning it to the data.
    cv::Mat mat = m_mat;

    data.meta[meta_tag::timestamp] = utils::timestamp<ts_t>();
    data.meta[meta_tag::seq_id] = m_curr_seq_id++;
    data = mat;
    m_next_tick_ts += m_latency;

    return true;
}

cv::GMetaArg DummySource::descr_of() const {
    return cv::GMetaArg{cv::descr_of(m_mat)};
}

void DummySource::setDropFrames(const bool drop_frames) {
    m_drop_frames = drop_frames;
}
