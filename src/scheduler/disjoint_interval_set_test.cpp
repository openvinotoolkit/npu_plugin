#include "gtest/gtest.h"
#include "disjoint_interval_set.hpp"

using namespace mv::lp_scheduler;

typedef Disjoint_Interval_Set<int, std::string> disjoint_interval_set_t;

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
