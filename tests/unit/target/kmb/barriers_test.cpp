#include "gtest/gtest.h"
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/target/kmb/barrier_definition.hpp"
#include "include/mcm/target/kmb/barrier_deps.hpp"
#include <set>
#include <string>

TEST(barrier, api)
{
    mv::Barrier b;

    b.setIndex(1);
    b.setGroup(2);
    b.addProducer("p1");
    b.addConsumer("c1");
    ASSERT_EQ(b.getIndex(), 1);
    ASSERT_EQ(b.getGroup(), 2);
    ASSERT_EQ(b.getNumProducers(), 1);
    ASSERT_EQ(b.getNumConsumers(), 1);
    ASSERT_TRUE(b.hasProducers());
    ASSERT_TRUE(b.hasConsumers());

    std::set<std::string> p = b.getProducers();
    std::set<std::string> pExpected;
    pExpected.insert("p1");
    ASSERT_EQ(p, pExpected);

    std::set<std::string> c = b.getConsumers();
    std::set<std::string> cExpected;
    cExpected.insert("c1");
    ASSERT_EQ(c, cExpected);

    b.addProducer("p2");
    b.addProducer("p3");
    pExpected.insert("p2");
    pExpected.insert("p3");
    ASSERT_EQ(b.getProducers(), pExpected);

    b.removeProducer("p1");
    ASSERT_EQ(b.getNumProducers(), 2);

    b.removeConsumer("c1");
    ASSERT_EQ(b.getNumConsumers(), 0);

    std::set<std::string> pList;
    pList.insert("p1");
    pList.insert("p2");
    pList.insert("p3");

    std::set<std::string> cList;
    cList.insert("c1");
    cList.insert("c2");

    mv::Barrier b2(pList, cList);
    ASSERT_EQ(b2.getNumProducers(), pList.size());
    ASSERT_EQ(b2.getNumConsumers(), cList.size());
    ASSERT_EQ(b2.getProducers(), pList);
    ASSERT_EQ(b2.getConsumers(), cList);

    b2.setNumProducers(2);
    b2.setNumConsumers(42);
    ASSERT_EQ(b2.getNumProducers(), 2);
    ASSERT_EQ(b2.getNumConsumers(), 42);

    b2.clear();
    ASSERT_EQ(b2.getNumProducers(), 0);
    ASSERT_EQ(b2.getNumProducers(), 0);
    ASSERT_EQ(b2.getGroup(), -1);
    ASSERT_EQ(b2.getIndex(), -1);
    ASSERT_FALSE(b2.hasProducers());
    ASSERT_FALSE(b2.hasConsumers());

}

TEST(barrier_deps, api)
{
    mv::BarrierDependencies bdep;
    bdep.setWaitBarrier(1);
    ASSERT_EQ(bdep.getWait(), 1);

    bdep.addUpdateBarrier(2);
    bdep.addUpdateBarrier(3);
    std::vector<unsigned> updateExpected = { 2, 3 };
    ASSERT_EQ(bdep.getUpdate(), updateExpected);

}

TEST(barrier, to_json)
{
    mv::Barrier b;

    b.setIndex(1);
    b.setGroup(2);
    b.addProducer("p1");
    b.addProducer("p2");
    b.addConsumer("c1");
    b.addConsumer("c2");

    mv::Attribute a(b);
    std::string expected = "{\"attrType\":\"Barrier\",\""
                            "content\":{\"consumers\":[\"c2\",\"c1\"],"
                            "\"group\":2,\"index\":1,"
                            "\"numConsumers\":2,\"numProducers\":2"
                            ",\"producers\":[\"p2\",\"p1\"]}}";

    ASSERT_EQ(a.toJSON().stringify(), expected);
}

TEST(barrier_deps, to_json)
{
    mv::BarrierDependencies bdep;
    bdep.setWaitBarrier(1);
    ASSERT_EQ(bdep.getWait(), 1);

    bdep.addUpdateBarrier(2);
    bdep.addUpdateBarrier(3);

    mv::Attribute a(bdep);
    std::string expected = "{\"attrType\":\"BarrierDependencies\","
                            "\"content\":{\"update\":[2,3],\"wait\":1}}";

    ASSERT_EQ(a.toJSON().stringify(), expected);
}
