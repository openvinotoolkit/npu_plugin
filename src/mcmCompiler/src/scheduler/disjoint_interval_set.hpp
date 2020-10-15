#ifndef DISJOINT_INTERVAL_SET_HPP
#define DISJOINT_INTERVAL_SET_HPP
#include <cassert>
#include <limits>
#include <map>
#include <unordered_set>

namespace mv {
namespace lp_scheduler {

template<typename T>
struct interval_traits {
  typedef int unit_t;
  typedef int interval_t;

  static unit_t interval_begin(const interval_t&);
  static unit_t interval_end(const interval_t&);
  static void set_interval(interval_t&, const unit_t& beg, const unit_t& end);
}; // struct interval_traits //



//NOTE: these are only for intervals on an integer lattice //
template<typename T>
struct Interval_Utils {
  typedef interval_traits<T> traits;
  typedef typename traits::interval_t interval_t;
  typedef typename traits::unit_t unit_t;

  static bool intersects(const unit_t& abeg, const unit_t& aend,
      const unit_t& bbeg, const unit_t& bend) {
    assert((abeg <= aend) && (bbeg <= bend));
    return std::max(abeg, bbeg) <= std::min(aend, bend);
  }

  static bool intersects(const interval_t& a, const interval_t& b) {
    unit_t abeg = traits::interval_begin(a);
    unit_t aend = traits::interval_end(a);

    unit_t bbeg = traits::interval_begin(b);
    unit_t bend = traits::interval_end(b);

    assert((abeg <= aend) && (bbeg <= bend));

    return std::max(abeg, bbeg) <= std::min(aend, bend);
  }

  static bool interval_intersection(const interval_t& a, const interval_t& b,
      interval_t& output) {

    unit_t abeg = traits::interval_begin(a);
    unit_t aend = traits::interval_end(a);

    unit_t bbeg = traits::interval_begin(b);
    unit_t bend = traits::interval_end(b);

    assert((abeg <= aend) && (bbeg <= bend));

    unit_t obeg = std::max(abeg, bbeg);
    unit_t oend = std::min(aend, bend);
    bool intersects = (obeg <= oend);


    if (intersects) {
      traits::set_interval(output, obeg, oend);
    }

    return intersects;
  }

  static bool interval_intersection(const interval_t& a, const interval_t& b,
      unit_t& obeg, unit_t& oend) {

    unit_t abeg = traits::interval_begin(a);
    unit_t aend = traits::interval_end(a);

    unit_t bbeg = traits::interval_begin(b);
    unit_t bend = traits::interval_end(b);

    assert((abeg <= aend) && (bbeg <= bend));

    obeg = std::max(abeg, bbeg);
    oend = std::min(aend, bend);
    return (obeg <= oend);
  }

  static bool interval_overlap_union(const interval_t& a, const interval_t& b,
      interval_t& result) {

    if (!intersects(a,b)) { return false; }

    unit_t abeg = traits::interval_begin(a);
    unit_t aend = traits::interval_end(a);

    unit_t bbeg = traits::interval_begin(b);
    unit_t bend = traits::interval_end(b);

    unit_t ubeg = std::min(abeg, bbeg);
    unit_t uend = std::max(aend, bend);

    traits::set_interval(result, ubeg, uend);
    return true;
  }

  // Checks if interval 'a' is subset of interval 'b' //
  static bool is_subset(const interval_t& a, const interval_t& b) {
    if (!intersects(a, b)) { return false; }

    unit_t abeg = traits::interval_begin(a);
    unit_t aend = traits::interval_end(a);

    unit_t bbeg = traits::interval_begin(b);
    unit_t bend = traits::interval_end(b);

    unit_t ibeg = std::max(abeg, bbeg);
    unit_t iend = std::min(aend, bend);

    return (ibeg == abeg) && (iend == aend);
  }

  static bool is_subset(unit_t abeg, unit_t aend, unit_t bbeg, unit_t bend) {
    assert((abeg <= aend) && (bbeg <= bend));

    if (std::max(abeg, bbeg) > std::min(aend, bend)) { return false; }
    unit_t ibeg = std::max(abeg, bbeg);
    unit_t iend = std::min(aend, bend);

    return (ibeg == abeg) && (iend == aend);
  }



  template<typename BackInsertIterator>
  static size_t interval_xor(const interval_t& a, const interval_t& b,
      BackInsertIterator out_itr) {
    size_t ret_size = 0UL;
    if (!intersects(a, b)) {
      *out_itr = a; ++out_itr;
      *out_itr = b; ++out_itr;
      ret_size = 2UL;
    } else {
      unit_t abeg = traits::interval_begin(a);
      unit_t aend = traits::interval_end(a);

      unit_t bbeg = traits::interval_begin(b);
      unit_t bend = traits::interval_end(b);

      assert((abeg <= aend) && (bbeg <= bend));

      unit_t obeg = std::max(abeg, bbeg);
      unit_t oend = std::min(aend, bend);
      unit_t ubeg = std::min(abeg, bbeg);
      unit_t uend = std::max(aend, bend);

      // [ubeg,obeg] and [oend,uend] //
      --obeg; // note: if its not integer lattice this is an open interval //
      if ((ubeg <= obeg) && (obeg < std::min(aend, bend))) {
        traits::set_interval(*out_itr, ubeg, obeg);
        ++out_itr; ++ret_size;
      }

      ++oend;
      if ((oend <= uend) && (std::max(abeg, bbeg) < oend)) {
        traits::set_interval(*out_itr, oend, uend);
        ++out_itr; ++ret_size;
      }
    }
    return ret_size;
  }

  static size_t interval_xor(const interval_t& a, const interval_t& b,
      unit_t (&out_begin)[2UL], unit_t (&out_end)[2UL]) {

    unit_t abeg = traits::interval_begin(a);
    unit_t aend = traits::interval_end(a);

    unit_t bbeg = traits::interval_begin(b);
    unit_t bend = traits::interval_end(b);

    assert((abeg <= aend) && (bbeg <= bend));

    size_t ret_size = 0UL;

    if (!intersects(a, b)) {
      out_begin[0] = abeg; out_end[0] = aend;
      out_begin[1] = bbeg; out_end[1] = bend;
      ret_size = 2UL;
    } else {

      unit_t obeg = std::max(abeg, bbeg);
      unit_t oend = std::min(aend, bend);
      unit_t ubeg = std::min(abeg, bbeg);
      unit_t uend = std::max(aend, bend);

      // [ubeg,obeg] and [oend,uend] //
      --obeg; // note: if its not integer lattice this is an open interval //
      if ((ubeg <= obeg) && (obeg < std::min(aend, bend))) {
        out_begin[ret_size] = ubeg; out_end[ret_size] = obeg;
        ++ret_size;
      }

      ++oend;
      if ((oend <= uend) && (std::max(abeg, bbeg) < oend)) {
        out_begin[ret_size] = oend; out_end[ret_size] = uend;
        ++ret_size;
      }
    }
    return ret_size;
  }

  // NOTE: the return value ensures out_end[0] < out_begin[1] //
  static size_t interval_xor(const unit_t& abeg, const unit_t& aend,
      const unit_t& bbeg, const unit_t& bend,
      unit_t (&out_begin)[2UL], unit_t (&out_end)[2UL]) {

    size_t ret_size = 0UL;

    if (!intersects(abeg, aend, bbeg, bend)) {
      out_begin[0] = abeg; out_end[0] = aend;
      out_begin[1] = bbeg; out_end[1] = bend;
      ret_size = 2UL;
    } else {

      unit_t obeg = std::max(abeg, bbeg);
      unit_t oend = std::min(aend, bend);
      unit_t ubeg = std::min(abeg, bbeg);
      unit_t uend = std::max(aend, bend);

      // [ubeg,obeg] and [oend,uend] //
      --obeg; // note: if its not integer lattice this is an open interval //
      if ((ubeg <= obeg) && (obeg < std::min(aend, bend))) {
        out_begin[ret_size] = ubeg; out_end[ret_size] = obeg;
        ++ret_size;
      }

      ++oend;
      if ((oend <= uend) && (std::max(abeg, bbeg) < oend)) {
        out_begin[ret_size] = oend; out_end[ret_size] = uend;
        ++ret_size;
      }
    }
    return ret_size;
  }

}; // struct Interval_Utils //


// Maintains a set of dynamic disjoint intervals and each interval is
// associated with a corresponding element. Supports the following operations:
// 
// range_query: input [a, b] (a <= b) reports all the overlapping elements in 
// the set which overlap [a,b]
//
// insert: inserts a interval [a,b] iff its disjoint with other elements in the
// set. Returns false if [a,b] is not disjoint.
//
// erase: erase an interval [a,b] from the data structure.
// 
// NOTE: disjoint means no overlap and no touch.
template<typename Unit, typename Element>
class Disjoint_Interval_Set {
  public:
    typedef Unit unit_t;
    typedef Element element_t;

    //            ----[*]---
    //                 x
    enum orient_e {
      LEFT_END= 0, RIGHT_END = 1
    }; // enum orient_e //

    struct end_point_t {
      unit_t x_;
      orient_e orient_;

      bool operator==(const end_point_t& o) const {
        return (x_ == o.x_) && (orient_ == o.orient_);
      }

      bool operator<(const end_point_t& o) const {
        return (x_ != o.x_) ? (x_ < o.x_) : orient_ < o.orient_;
      }

      bool is_left_end() const { return orient_ == LEFT_END; }
      bool is_right_end() const { return orient_ == RIGHT_END; }

      end_point_t(unit_t x, orient_e orient) : x_(x), orient_(orient) {}
    }; // struct end_point_t //

    typedef std::map<end_point_t, element_t> end_point_tree_t;
    typedef typename end_point_tree_t::const_iterator end_point_iterator_t;
    typedef typename end_point_tree_t::const_iterator
        const_end_point_iterator_t;

    //Invariant: itr_begin_ should always point to 
    class interval_iterator_t {
      public:

        interval_iterator_t(const_end_point_iterator_t begin,
            const_end_point_iterator_t end)
          : itr_begin_(begin), itr_end_(end) {}

        interval_iterator_t(): itr_begin_(), itr_end_() {}

        // only invalid iterators are equivalent //
        bool operator==(const interval_iterator_t& o) const {
          return (itr_begin_ == itr_end_) && (o.itr_begin_ == o.itr_end_);
        }
        bool operator!=(const interval_iterator_t& o) const {
          return !(*this == o);
        }

        // Precondition: itr_begin_ != itr_end_ //
        const interval_iterator_t& operator++() {
          // Invariant: itr_begin_ always points to a left end //
          ++itr_begin_;
          assert(itr_begin_ != itr_end_);
          ++itr_begin_;
          return *this;
        }

        // Precondition: itr_begin_ != itr_end_ //
        const element_t& operator*() const { return itr_begin_->second; }

        // Precondition: itr_begin_ != itr_end_ //
        unit_t interval_begin() const { return (itr_begin_->first).x_; }

        // Precondition: itr_begin_ != itr_end_ //
        unit_t interval_end() const {
          const_end_point_iterator_t itr_next = itr_begin_; ++itr_next;
          return (itr_next->first).x_;
        }

      private:
        const_end_point_iterator_t itr_begin_;
        const_end_point_iterator_t itr_end_;
    }; // class range_query_iterator_t //

    // Iterator over free intervals in the data structure //
    class free_interval_iterator_t {
      public:

        free_interval_iterator_t(
            const_end_point_iterator_t begin, const_end_point_iterator_t end,
            unit_t prev_right=std::numeric_limits<unit_t>::min())
        : itr_begin_(begin), itr_end_(end), prev_right_end_(prev_right) {}

        // this constructor creates an invalid instance //
        free_interval_iterator_t() : itr_begin_(), itr_end_(),
          prev_right_end_(std::numeric_limits<unit_t>::max()) {}

        // only invalid iterators are equivalent //
        bool operator==(const free_interval_iterator_t& o) const {
          return invalid() && o.invalid();
        }

        bool operator!=(const free_interval_iterator_t& o) const {
          return !(*this == o);
        }

        //WARNING: please keep in mind the interval is open on both
        //ends. That is its of the following form:
        // { x | interval_begin() < x < interval_end() }
        unit_t interval_begin() const { return prev_right_end_; }
        unit_t interval_end() const {
          return (itr_begin_ == itr_end_) ? std::numeric_limits<unit_t>::max() :
            (itr_begin_->first).x_;
        }

        // Precondition: !invalid() //
        const free_interval_iterator_t& operator++() {
          assert(!invalid());
          if (itr_begin_ != itr_end_) {
            // Invariant: itr_begin_ always points to the left end in the DS //
            ++itr_begin_;
            prev_right_end_ = (itr_begin_->first).x_;
            assert((itr_begin_ != itr_end_));
            ++itr_begin_;
          } else {
            invalidate(); 
          }
          return *this;
        }

      private:

        void invalidate() {
          prev_right_end_ = std::numeric_limits<unit_t>::max();
        }

        bool invalid() const { 
          return (prev_right_end_ == std::numeric_limits<unit_t>::max());
        }

        const_end_point_iterator_t itr_begin_;
        const_end_point_iterator_t itr_end_;
        unit_t prev_right_end_;
    }; // class free_interval_iterator_t //


    bool insert(const unit_t& ibeg, const unit_t& iend, const element_t& e) {
      assert(ibeg <= iend);
      end_point_t lkey(ibeg, LEFT_END), rkey(iend, RIGHT_END);
      end_point_iterator_t itr = tree_.lower_bound(lkey);

      if (!is_interval_disjoint(lkey, rkey, itr)) { return false; }
      tree_.insert(itr /*hint*/, std::make_pair(lkey, e));
      tree_.insert(itr /*hint*/, std::make_pair(rkey, e));
      return true;
    }

    bool erase(const unit_t& ibeg, const unit_t& iend) {
      end_point_iterator_t litr = tree_.find(end_point_t(ibeg, LEFT_END));
      end_point_iterator_t ritr = tree_.find(end_point_t(iend, RIGHT_END));
      if ((litr == tree_.end()) || (ritr == tree_.end())) { return false; }

      tree_.erase(litr);
      tree_.erase(ritr);
      return true;
    }


    bool overlaps(const unit_t& ibeg, const unit_t& iend) const {
      assert(ibeg <= iend);
      end_point_t lkey(ibeg, LEFT_END), rkey(iend, RIGHT_END);

      return !is_interval_disjoint(lkey, rkey, tree_.lower_bound(lkey));
    }

    interval_iterator_t query(const unit_t& ibeg, const unit_t& iend) const {
      const_end_point_iterator_t litr =
          tree_.lower_bound(end_point_t(ibeg, LEFT_END));

      if ( (litr != tree_.end()) && ((litr->first).is_right_end()) ) {
        --litr;
        assert(litr != tree_.end());
      }

      const_end_point_iterator_t ritr =
        tree_.lower_bound(end_point_t(iend, RIGHT_END));
      if ( (ritr != tree_.end()) && ((ritr->first).is_right_end()) ) {
        ++ritr;
      }
      return interval_iterator_t(litr, ritr);
    }

    interval_iterator_t begin() const {
      return interval_iterator_t(tree_.begin(), tree_.end());
    }

    interval_iterator_t end() const {
      return interval_iterator_t(tree_.end(), tree_.end());
    }

    // NOTE: the number of free intervals is n+1 if there are n intervals
    // in this data structure. Even if the data structure is empty there is
    // a free interval. The iterator returns open intervals of the form (a,b)
    // not including 'a' and 'b'.
    free_interval_iterator_t begin_free_intervals() const {
      return free_interval_iterator_t(tree_.begin(), tree_.end());
    }

    free_interval_iterator_t end_free_intervals() const {
      return free_interval_iterator_t();
    }

    // static utility method //
    static bool do_intervals_overlap(unit_t abeg, unit_t aend,
        unit_t bbeg, unit_t bend) {
      return std::max(abeg, bbeg) <= std::min(aend, bend);
    }

    Disjoint_Interval_Set() : tree_() {}

    size_t size() const { assert(tree_.size()%2 == 0); return tree_.size()/2; }
    bool empty() const { return tree_.empty(); }
    void clear() { tree_.clear(); }

  private:

    bool is_left_end(const_end_point_iterator_t& o) const {
      return (o->first).is_left_end();
    }

    bool is_interval_disjoint(const end_point_t& lkey, const end_point_t& rkey,
        const_end_point_iterator_t lkey_lower_bound) const {

      if (lkey_lower_bound == tree_.end()) { return true; }
      const end_point_t &lkey_lb = lkey_lower_bound->first;
      return (lkey_lb.is_left_end() && (lkey.x_ < lkey_lb.x_)
          && (rkey.x_ < lkey_lb.x_));
    }

    ////////////////////////////////////////////////////////////////////////////
    end_point_tree_t tree_; // balanced search tree on end_point_t //
}; // class Disjoint_Interval_Set //

} // namespace lp_scheduler //
} // namespace mv

#endif
