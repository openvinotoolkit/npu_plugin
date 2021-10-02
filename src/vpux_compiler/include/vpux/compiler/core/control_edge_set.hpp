#pragma once

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

// #include "include/mcm/computation/model/base_op_model.hpp"
// #include "include/mcm/computation/model/control_model.hpp"
// #include "include/mcm/computation/model/data_model.hpp"
// #include "include/mcm/computation/model/iterator/tensor.hpp"
// #include "include/mcm/logger/logger.hpp"
// #include "pass/lp_scheduler/control_edge_generator.hpp"
// #include "pass/lp_scheduler/operation_precedence_dag.hpp"
// #include "scheduler/dag_address_generator.hpp"
// #include "include/mcm/utils/warning_manager.hpp"
// #include "include/mcm/utils/hash.hpp"

#include "vpux/compiler/core/control_edge_generator.hpp"
#include "vpux/compiler/core/operation_precedence_dag.hpp"

namespace vpux {

struct Control_Edge {
    Control_Edge(size_t source, size_t sink): source_(source), sink_(sink) {
    }

    bool operator<(const Control_Edge& o) const {
        return (source_ != o.source_) ? (source_ < o.source_) : (sink_ < o.sink_);
    }

    std::string source_name() const {
        return std::to_string((int)source_);
    }
    std::string sink_name() const {
        return std::to_string((int)sink_);
    }

    size_t source_;
    size_t sink_;
};  // struct Control_Edge //

template <>
struct interval_traits<Scheduled_Op> {
    typedef size_t unit_t;
    typedef Scheduled_Op interval_t;

    static unit_t interval_begin(const interval_t& interval) {
        return interval.cmx_address_start_;
    }

    static unit_t interval_end(const interval_t& interval) {
        return interval.cmx_address_end_;
    }
};  // struct interval_traits<Scheduled_Op> //

class Control_Edge_Set {
public:
    typedef size_t operation_t;
    typedef Scheduled_Op scheduled_op_t;
    typedef Control_Edge control_edge_t;
    typedef std::set<control_edge_t> edge_set_t;
    typedef typename edge_set_t::const_iterator const_edge_iterator_t;

    Control_Edge_Set(): control_edge_set_() {
    }

    void operator()(const scheduled_op_t& a, const scheduled_op_t& b) {
        control_edge_set_.insert(control_edge_t(a.op_, b.op_));
    }

    const_edge_iterator_t begin() const {
        return control_edge_set_.begin();
    }
    const_edge_iterator_t end() const {
        return control_edge_set_.end();
    }

private:
    std::set<control_edge_t> control_edge_set_;
};  //  class Control_Edge_Set //

}  // namespace vpux
