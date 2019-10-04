#ifndef DISJOINT_INTERVAL_SET_HPP
#define DISJOINT_INTERVAL_SET_HPP
#include <map>
#include <unordered_set>

namespace mv {
namespace lp_scheduler {

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

    interval_iterator_t end() const {
      return interval_iterator_t(tree_.end(), tree_.end());
    }

    // static utility method //
    static bool do_intervals_overlap(unit_t abeg, unit_t aend,
        unit_t bbeg, unit_t bend) {
      return std::max(abeg, bbeg) <= std::min(aend, bend);
    }

    Disjoint_Interval_Set() : tree_() {}

    size_t size() const { assert(tree_.size()%2 == 0); return tree_.size()/2; }
    bool empty() const { return tree_.empty(); }

  private:

    bool is_left_end(const_end_point_iterator_t& o) const {
      return (o->first).is_left_end();
    }

    bool is_interval_disjoint(const end_point_t& lkey, const end_point_t& rkey,
        const_end_point_iterator_t lkey_lower_bound) const {

      const end_point_t &lkey_lb = lkey_lower_bound->first;

      return (lkey_lower_bound == tree_.end()) || (lkey_lb.is_left_end() &&
          (lkey.x_ < lkey_lb.x_) && (rkey.x_ < lkey_lb.x_));
    }


    end_point_tree_t tree_; // balanced search tree on end_point_t //
}; // class Disjoint_Interval_Set //

} // namespace lp_scheduler //
} // namespace mv

#endif
