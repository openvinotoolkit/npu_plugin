#ifndef SCHEDULER_UNIT_TEST_UTILS_HPP
#define SCHEDULER_UNIT_TEST_UTILS_HPP

#include <iostream>

namespace mv_unit_tests {

struct interval_t {
  interval_t(int b, int e, const std::string& id="")
    : beg_(b), end_(e), id_(id) {}

  interval_t() : beg_(), end_(), id_() {}

  bool operator==(const interval_t& o) const {
    return (beg_ == o.beg_) && (end_ == o.end_) && (id_ == o.id_);
  }

  void print() const {
    std::cout << "[ " << beg_ << " " << end_ << " " << id_ << "]" << std::endl;
  }

  int beg_;
  int end_;
  std::string id_;
}; // struct interval_t //

} // namespace mv_unit_tests //

namespace mv {
namespace lp_scheduler {

template<>
struct interval_traits<mv_unit_tests::interval_t> {
  typedef int unit_t;
  typedef mv_unit_tests::interval_t interval_t;



  static unit_t interval_begin(const interval_t& interval) {
    return interval.beg_;
  }

  static unit_t interval_end(const interval_t& interval) {
    return interval.end_;
  }

  static void set_interval(interval_t& interval,
        const unit_t& beg, const unit_t& end) {
    interval.beg_ = beg; 
    interval.end_ = end;
  }

}; // struct interval_traits //

} // namespace lp_scheduler //
} // namespace mv //



#endif
