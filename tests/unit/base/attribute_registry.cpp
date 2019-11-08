#include <algorithm>
#include <typeinfo>
#include <typeindex>
#include "gtest/gtest.h"
#include "include/mcm/base/attribute_registry.hpp"
#include "include/mcm/base/attribute.hpp"

struct TestAttr
{

    std::string str;

};

#define TO_STR(x) #x

static std::string testString = "TestValue";
static std::string typeName = TO_STR(TestAttr);
static std::string toJSONPostfix = "_toJSON";
static std::string fromJSONPostfix = "_fromJSON";
static std::string toStringPostfix = "_toString";
static std::string testDescription = "TestDescription";
static std::type_index typeIdx = typeid(TestAttr);

static void setAttrReg()
{

    static std::function<mv::json::Value(const mv::Attribute&)> toJSON = 
        [](const mv::Attribute& a)->mv::json::Value
    {   
        return mv::json::Value(a.get<TestAttr>().str + toJSONPostfix);
    };

    static std::function<mv::Attribute(const mv::json::Value& v)> fromJSON = 
        [](const mv::json::Value& v)->mv::Attribute
    {   
        return TestAttr({v.get<std::string>() + fromJSONPostfix});
    };

    static std::function<std::string(const mv::Attribute&)> toString = 
        [](const mv::Attribute& a)->std::string
    {   
        return a.get<TestAttr>().str + toStringPostfix;
    };


    mv::attr::AttributeRegistry::instance().enter<TestAttr>()
        .setName(typeName)
        .setDescription(testDescription)
        .setToJSONFunc(toJSON)
        .setFromJSONFunc(fromJSON)
        .setToStringFunc(toString);

}

static void resetAttrReg()
{
    mv::attr::AttributeRegistry::instance().remove(typeid(TestAttr));
}

TEST(attr_registry, add_remove_entry)
{
    std::size_t initSize = mv::attr::AttributeRegistry::instance().size();
    setAttrReg();
    auto attrList = mv::attr::AttributeRegistry::instance().list();
    ASSERT_TRUE(std::find(attrList.begin(), attrList.end(), typeid(TestAttr)) != attrList.end());
    resetAttrReg();
    ASSERT_EQ(initSize, mv::attr::AttributeRegistry::instance().size());

}

TEST(attr_registry, get_type_id)
{

    setAttrReg();
    ASSERT_EQ(typeIdx, mv::attr::AttributeRegistry::instance().getTypeID(typeName));
    resetAttrReg();

}


TEST(attr_registry, get_type_name)
{

    setAttrReg();
    ASSERT_EQ(typeName, mv::attr::AttributeRegistry::instance().getTypeName(typeid(TestAttr)));
    resetAttrReg();

}

TEST(attr_registry, get_to_json)
{

    setAttrReg();

    mv::Attribute a1 = TestAttr({testString});
    mv::json::Value v1 = testString + toJSONPostfix;

    ASSERT_EQ(v1.get<std::string>(), mv::attr::AttributeRegistry::instance().getToJSONFunc(typeid(TestAttr))(a1).get<std::string>());

    resetAttrReg();

}

TEST(attr_registry, get_from_json)
{

    setAttrReg();

    mv::json::Value v1 = testString;

    ASSERT_EQ(v1.get<std::string>() + fromJSONPostfix, 
        mv::attr::AttributeRegistry::instance().getFromJSONFunc(typeid(TestAttr))(v1).get<TestAttr>().str);

    resetAttrReg();

}

TEST(attr_registry, get_to_string)
{

    setAttrReg();

    mv::Attribute a1 = TestAttr({testString});

    ASSERT_EQ(testString + toStringPostfix, 
        mv::attr::AttributeRegistry::instance().getToStringFunc(typeid(TestAttr))(a1));

    resetAttrReg();

}