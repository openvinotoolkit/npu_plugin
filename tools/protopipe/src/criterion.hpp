//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

struct ITermCriterion {
    using Ptr = std::shared_ptr<ITermCriterion>;
    virtual void init() = 0;
    virtual void update() = 0;
    virtual bool check() const = 0;
};

class Iterations : public ITermCriterion {
public:
    Iterations(size_t num_iters);

    void init() override;
    void update() override;
    bool check() const override;

private:
    size_t m_num_iters;
    size_t m_counter;
};

class TimeOut : public ITermCriterion {
public:
    TimeOut(size_t time_in_us);

    void init() override;
    void update() override;
    bool check() const override;

private:
    size_t m_time_in_us;
    size_t m_start_ts;
};
