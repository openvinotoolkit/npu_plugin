#include "gtest/gtest.h"

#include "disjoint_interval_set.hpp"
#include "scheduler/scheduler_unit_test_utils.hpp"

using namespace mv::lp_scheduler;

typedef int unit_t;
typedef Disjoint_Interval_Set<unit_t, std::string> disjoint_interval_set_t;

TEST(Disjoint_Interval_Set, test_disjoint_dont_allow_touch) {
  disjoint_interval_set_t dset;

  EXPECT_FALSE(dset.overlaps(1,10));
  EXPECT_TRUE(dset.insert(1,10, "interval1"));
  EXPECT_FALSE(dset.insert(1,10, "interval2")); // same interval //

  EXPECT_FALSE(dset.insert(-100,1, "touch_from_left"));
  EXPECT_FALSE(dset.insert(10,100, "touch_from_right"));
  EXPECT_FALSE(dset.insert(5,8, "fully_inside"));
  EXPECT_FALSE(dset.insert(8,100, "cross_right"));
  EXPECT_FALSE(dset.insert(-100,2, "cross_left"));

  EXPECT_TRUE(dset.insert(20,20, "point"));
  EXPECT_FALSE(dset.insert(12,50, "overlaps"));
  EXPECT_FALSE(dset.insert(20,30, "touch_right"));
  EXPECT_FALSE(dset.insert(15,20, "touch_left"));

}

template<typename DataStructure, typename OutputContainer>
bool QueryAndGetResults(const DataStructure& data_structure,
    typename DataStructure::unit_t ibeg, typename DataStructure::unit_t iend,
    OutputContainer& result) {
  result.clear();
  for (auto itr=data_structure.query(ibeg, iend); itr!=data_structure.end();
      ++itr) {
    if (!DataStructure::do_intervals_overlap(ibeg, iend,
          itr.interval_begin(), itr.interval_end())) { return false; }
    result.push_back(*itr);
  }
  return true;
}


TEST(Disjoint_Interval_Set, range_query_iterator) {

  disjoint_interval_set_t dset;

  EXPECT_TRUE(dset.insert(10,10, "interval1"));
  EXPECT_TRUE(dset.insert(12,12, "interval2"));

  EXPECT_FALSE(dset.insert(10,12, "interval3"));
  EXPECT_FALSE(dset.insert(12,13, "interval4"));
  EXPECT_FALSE(dset.insert(1,15, "interval4"));

  EXPECT_TRUE(dset.insert(13, 20, "interval3"));

  // [10]  [12] [13,20] //

  { // query: [-100,100] //
    std::vector<std::string> expected = {"interval1", "interval2", "interval3"};
    std::vector<std::string> found;
    ASSERT_TRUE( QueryAndGetResults(dset, -100, 100, found) );
    EXPECT_EQ(found, expected);
  }

  { // query: [11,12] //
    std::vector<std::string> expected = {"interval2"};
    std::vector<std::string> found;

    ASSERT_TRUE( QueryAndGetResults(dset, 11, 12, found) );
    EXPECT_EQ(found, expected);
  }

  {
    // point query: [11,11] //
    std::vector<std::string> found;

    ASSERT_TRUE( QueryAndGetResults(dset, 11, 11, found) );
    EXPECT_TRUE(found.empty());

    ASSERT_TRUE( QueryAndGetResults(dset, -100, 9, found) );
    EXPECT_TRUE(found.empty());

    ASSERT_TRUE( QueryAndGetResults(dset, 21, 100, found) );
    EXPECT_TRUE(found.empty());
  }

  {
    // query: [10,10] //
    std::vector<std::string> expected = {"interval1"};
    std::vector<std::string> found;

    ASSERT_TRUE( QueryAndGetResults(dset, 10, 10, found) );
    EXPECT_EQ(found, expected);
  }
  
}

TEST(Disjoint_Interval_Set, insert_and_erase) {

  disjoint_interval_set_t dset;

  EXPECT_TRUE(dset.insert(10,10, "interval1"));
  EXPECT_TRUE(dset.insert(12,12, "interval2"));
  EXPECT_TRUE(dset.insert(15, 20, "interval3"));

  // [10]  [12] [15,20] //

  {
    dset.erase(10,10);
    std::vector<std::string> expected = {"interval2", "interval3"};
    std::vector<std::string> found;
    ASSERT_TRUE( QueryAndGetResults(dset, -100, 100, found) );
    EXPECT_EQ(found, expected);
    EXPECT_TRUE(dset.insert(10, 10, "interval1"));
  }

  {
    dset.erase(15,20);
    std::vector<std::string> expected = {"interval1", "interval2"};
    std::vector<std::string> found;
    ASSERT_TRUE( QueryAndGetResults(dset, -100, 100, found) );
    EXPECT_EQ(found, expected);
    EXPECT_TRUE(dset.insert(15, 20, "interval3"));
  }

  {
    dset.erase(12,12);
    std::vector<std::string> expected = {"interval1", "interval3"};
    std::vector<std::string> found;
    ASSERT_TRUE( QueryAndGetResults(dset, -100, 100, found) );
    EXPECT_EQ(found, expected);
    EXPECT_TRUE(dset.insert(12, 12, "interval2"));
  }

}

TEST(Disjoint_Interval_Set, invalid_erase) {
  disjoint_interval_set_t dset;

  EXPECT_TRUE(dset.insert(10,10, "interval1"));
  EXPECT_TRUE(dset.insert(12,12, "interval2"));
  EXPECT_TRUE(dset.insert(15, 20, "interval3"));


  EXPECT_FALSE(dset.erase(8, 9));
  EXPECT_FALSE(dset.erase(10,11));
  EXPECT_FALSE(dset.erase(14,15));
  EXPECT_FALSE(dset.erase(12,13));

  EXPECT_EQ(dset.size(), 3UL);

  EXPECT_TRUE(dset.erase(12,12));
  EXPECT_EQ(dset.size(), 2UL);

  EXPECT_TRUE(dset.erase(15,20));
  EXPECT_EQ(dset.size(), 1UL);

  EXPECT_TRUE(dset.erase(10,10));
  EXPECT_TRUE(dset.empty());
}

TEST(Disjoint_Interval_Set, free_iterator_test_empty) {
  disjoint_interval_set_t dset;

  disjoint_interval_set_t::free_interval_iterator_t itr_begin, itr_end;

  itr_begin = dset.begin_free_intervals();
  itr_end = dset.end_free_intervals();

  // since set is empty the entire range is free : (-\infty, +\infty) //
  ASSERT_FALSE(itr_begin == itr_end);

  ASSERT_EQ(itr_begin.interval_begin(), std::numeric_limits<unit_t>::min());
  ASSERT_EQ(itr_begin.interval_end(), std::numeric_limits<unit_t>::max());

  ++itr_begin;
  ASSERT_TRUE(itr_begin == itr_end);
}

TEST(Disjoint_Interval, open_intervals_with_no_integral_points) {
  disjoint_interval_set_t dset;

  EXPECT_TRUE(dset.insert(10,10, "interval1"));
  EXPECT_TRUE(dset.insert(11,11, "interval2"));

  disjoint_interval_set_t::free_interval_iterator_t itr_begin, itr_end;
  
  itr_begin = dset.begin_free_intervals();
  itr_end = dset.end_free_intervals();

  ASSERT_FALSE(itr_begin == itr_end);
  ASSERT_EQ(itr_begin.interval_begin(), std::numeric_limits<unit_t>::min());
  ASSERT_EQ(itr_begin.interval_end(), 10);

  ++itr_begin;

  ASSERT_FALSE(itr_begin == itr_end);
  ASSERT_EQ(itr_begin.interval_begin(), 10);
  ASSERT_EQ(itr_begin.interval_end(), 11);
  // note a unit-length interval with integral end-points has no integers
  // inside the interval.

  ++itr_begin;
  ASSERT_FALSE(itr_begin == itr_end);
  ASSERT_EQ(itr_begin.interval_begin(), 11);
  ASSERT_EQ(itr_begin.interval_end(), std::numeric_limits<unit_t>::max());

  ++itr_begin;
  ASSERT_TRUE(itr_begin == itr_end);
}

TEST(Disjoint_Interval, open_intervals_with_typical_case) {
  disjoint_interval_set_t dset;

  EXPECT_TRUE(dset.insert(10,15, "interval1"));
  EXPECT_TRUE(dset.insert(20,30, "interval2"));

  disjoint_interval_set_t::free_interval_iterator_t itr_begin, itr_end;
  
  itr_begin = dset.begin_free_intervals();
  itr_end = dset.end_free_intervals();

  ASSERT_FALSE(itr_begin == itr_end);
  ASSERT_EQ(itr_begin.interval_begin(), std::numeric_limits<unit_t>::min());
  ASSERT_EQ(itr_begin.interval_end(), 10);

  ++itr_begin;

  ASSERT_FALSE(itr_begin == itr_end);
  ASSERT_EQ(itr_begin.interval_begin(), 15);
  ASSERT_EQ(itr_begin.interval_end(), 20);

  ++itr_begin;
  ASSERT_FALSE(itr_begin == itr_end);
  ASSERT_EQ(itr_begin.interval_begin(), 30);
  ASSERT_EQ(itr_begin.interval_end(), std::numeric_limits<unit_t>::max());

  ++itr_begin;
  ASSERT_TRUE(itr_begin == itr_end);
}


typedef mv_unit_tests::interval_t interval_t;
typedef mv::lp_scheduler::Interval_Utils<interval_t> interval_utils_t;

TEST(interval_utils, intersects) {
  interval_t a(1, 8), b(10, 12), c(3,7), d(8,11);

  EXPECT_FALSE(interval_utils_t::intersects(a, b)); 
  EXPECT_FALSE(interval_utils_t::intersects(b, a)); 

  EXPECT_TRUE(interval_utils_t::intersects(a, d));
  EXPECT_TRUE(interval_utils_t::intersects(d, a));

  EXPECT_TRUE(interval_utils_t::intersects(a, c));
  EXPECT_TRUE(interval_utils_t::intersects(c, a));

  EXPECT_TRUE(interval_utils_t::intersects(b, d));
  EXPECT_TRUE(interval_utils_t::intersects(d, b));

  EXPECT_TRUE(interval_utils_t::intersects(a, a));
  EXPECT_TRUE(interval_utils_t::intersects(b, b));
  EXPECT_TRUE(interval_utils_t::intersects(c, c));

}

TEST(interval_utils, interval_intersection) {
  interval_t a(1, 8), b(10, 12), c(3,7), d(8,11);
  interval_t result;

  EXPECT_FALSE(interval_utils_t::interval_intersection(a, b, result)); 
  EXPECT_FALSE(interval_utils_t::interval_intersection(b, a, result)); 

  EXPECT_TRUE(interval_utils_t::interval_intersection(a, d, result));
  EXPECT_EQ(result, interval_t(8, 8));

  result = interval_t(0,0);
  EXPECT_TRUE(interval_utils_t::interval_intersection(d, a, result));
  EXPECT_EQ(result, interval_t(8, 8));

  result = interval_t(0,0);
  EXPECT_TRUE(interval_utils_t::interval_intersection(a, c, result));
  EXPECT_EQ(result, interval_t(3, 7));

  result = interval_t(0,0);
  EXPECT_TRUE(interval_utils_t::interval_intersection(c, a, result));
  EXPECT_EQ(result, interval_t(3, 7));

  result = interval_t(0,0);
  EXPECT_TRUE(interval_utils_t::interval_intersection(b, d, result));
  EXPECT_EQ(result, interval_t(10, 11));

  result = interval_t(0,0);
  EXPECT_TRUE(interval_utils_t::interval_intersection(d, b, result));
  EXPECT_EQ(result, interval_t(10, 11));
}


TEST(interval_utils, interval_overlap_union) {
  interval_t a(1, 8), b(10, 12), c(3,7), d(8,11);
  interval_t result;

  EXPECT_FALSE(interval_utils_t::interval_overlap_union(a, b, result)); 
  EXPECT_FALSE(interval_utils_t::interval_overlap_union(b, a, result)); 

  EXPECT_TRUE(interval_utils_t::interval_overlap_union(a, d, result));
  EXPECT_EQ(result, interval_t(1, 11));

  result = interval_t(0,0);
  EXPECT_TRUE(interval_utils_t::interval_overlap_union(d, a, result));
  EXPECT_EQ(result, interval_t(1, 11));

  result = interval_t(0,0);
  EXPECT_TRUE(interval_utils_t::interval_overlap_union(a, c, result));
  EXPECT_EQ(result, interval_t(1, 8));

  result = interval_t(0,0);
  EXPECT_TRUE(interval_utils_t::interval_overlap_union(c, a, result));
  EXPECT_EQ(result, interval_t(1, 8));

  result = interval_t(0,0);
  EXPECT_TRUE(interval_utils_t::interval_overlap_union(b, d, result));
  EXPECT_EQ(result, interval_t(8, 12));

  result = interval_t(0,0);
  EXPECT_TRUE(interval_utils_t::interval_overlap_union(d, b, result));
  EXPECT_EQ(result, interval_t(8, 12));
}

TEST(interval_utils, is_subset) {
  EXPECT_FALSE(interval_utils_t::is_subset(interval_t(1,10), interval_t(3,8)));
  EXPECT_TRUE(interval_utils_t::is_subset(interval_t(3,8), interval_t(1,10)));

  EXPECT_TRUE(interval_utils_t::is_subset(interval_t(1,10), interval_t(1,10))); 
  EXPECT_FALSE(interval_utils_t::is_subset(interval_t(1,10),
          interval_t(10,20)));
  EXPECT_FALSE(interval_utils_t::is_subset(interval_t(1,10),
          interval_t(15,20)));
}


TEST(interval_utils, interval_xor) {
  interval_t result[2UL];

  EXPECT_EQ(interval_utils_t::interval_xor(interval_t(1,5), interval_t(10,20),
          result), 2UL);
  EXPECT_EQ(result[0], interval_t(1,5));
  EXPECT_EQ(result[1], interval_t(10,20));

  EXPECT_EQ(interval_utils_t::interval_xor(interval_t(1,15), interval_t(10,20),
        result), 2UL);
  EXPECT_EQ(result[0], interval_t(1,9));
  EXPECT_EQ(result[1], interval_t(16,20));


  EXPECT_EQ(interval_utils_t::interval_xor(interval_t(1,15), interval_t(10,15),
        result), 1UL);
  EXPECT_EQ(result[0], interval_t(1,9));

  EXPECT_EQ(interval_utils_t::interval_xor(interval_t(1,15), interval_t(1,15),
        result), 0UL);
}

TEST(interval_utils, interval_xor_overflow) {
  interval_t result[2UL];
  int int_max = std::numeric_limits<int>::max();
  int int_min = std::numeric_limits<int>::min();

  EXPECT_EQ(interval_utils_t::interval_xor(interval_t(int_min, int_max),
        interval_t(int_min, int_max), result), 0UL);


  EXPECT_EQ(interval_utils_t::interval_xor(interval_t((int_max-10),int_max),
        interval_t(int_max-8,int_max), result), 1UL);
  EXPECT_EQ(result[0], interval_t(int_max-10,int_max-9));

  EXPECT_EQ(interval_utils_t::interval_xor(interval_t(int_min,int_min+10),
        interval_t(int_min,int_min+5), result), 1UL);
  EXPECT_EQ(result[0], interval_t(int_min+6,int_min+10));
}


TEST(interval_utils, interval_xor_overflow_array) {
  int result_beg[2UL], result_end[2UL];
  int int_max = std::numeric_limits<int>::max();
  int int_min = std::numeric_limits<int>::min();

  EXPECT_EQ(interval_utils_t::interval_xor(interval_t(int_min, int_max),
        interval_t(int_min, int_max), result_beg, result_end), 0UL);


  EXPECT_EQ(interval_utils_t::interval_xor(interval_t((int_max-10),int_max),
        interval_t(int_max-8,int_max), result_beg, result_end), 1UL);
  EXPECT_EQ(result_beg[0], int_max-10); EXPECT_EQ(result_end[0], int_max-9);

  EXPECT_EQ(interval_utils_t::interval_xor(interval_t(int_min,int_min+10),
        interval_t(int_min,int_min+5), result_beg, result_end), 1UL);
  EXPECT_EQ(result_beg[0], int_min+6); EXPECT_EQ(result_end[0], int_min+10);

  EXPECT_EQ(interval_utils_t::interval_xor(interval_t(int_min+1,int_min+10),
        interval_t(int_min+2,int_min+5), result_beg, result_end), 2UL);

  EXPECT_EQ(result_beg[0], int_min+1); EXPECT_EQ(result_end[0], int_min+1);
  EXPECT_EQ(result_beg[1], int_min+6); EXPECT_EQ(result_end[1], int_min+10);
}


