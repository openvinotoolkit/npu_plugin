//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "criterion.hpp"
#include "utils.hpp"

#include <chrono>

Iterations::Iterations(size_t num_iters): m_num_iters(num_iters), m_counter(0) {
}

bool Iterations::check() const {
    return m_counter != m_num_iters;
}

void Iterations::update() {
    ++m_counter;
}

void Iterations::init() {
    m_counter = 0;
}

TimeOut::TimeOut(size_t time_in_us): m_time_in_us(time_in_us), m_start_ts(-1) {
}

bool TimeOut::check() const {
    return utils::timestamp<std::chrono::microseconds>() - m_start_ts < m_time_in_us;
}

void TimeOut::update(){/* do nothing */};

void TimeOut::init() {
    m_start_ts = utils::timestamp<std::chrono::microseconds>();
}
