#include "gtest/gtest.h"
#include "include/mcm/base/element.hpp"
#include "include/mcm/tensor/dtype.hpp"
#include "include/mcm/tensor/order.hpp"
#include "include/mcm/tensor/shape.hpp"

static bool vBool = true;
static double vDouble = 1.0;
static mv::DType vDType(mv::DTypeType::Float16);
static int vInt = 2;
static mv::Order vOrder(mv::OrderType::ColumnMajor);
static mv::Shape vShape({1, 2, 3});
static std::array<unsigned short, 2> vStdArrUnsignedShort2({4, 5});
static std::array<unsigned short, 3> vStdArrUnsignedShort3({6, 7, 8});
static std::array<unsigned short, 4> vStdArrUnsignedShort4({9, 10, 11, 12});
static std::size_t vStdSizeT = 3;
static std::string vStdString = "str";
static std::vector<std::size_t> vVecStdSizeT({13, 14, 15, 16, 17});
static unsigned short vUnsignedShort = 4;

static void setAllAttrTypes(mv::Element& e)
{
    e.clear();
    e.set<bool>("aBool", vBool);
    e.set<double>("aDouble", vDouble);
    e.set<mv::DType>("aDType", vDType);
    e.set<int>("aInt", vInt);
    e.set<mv::Order>("aOrder", vOrder);
    e.set<mv::Shape>("aShape", vShape);
    e.set<std::array<unsigned short, 2>>("aStdArrUnsignedShort2", vStdArrUnsignedShort2);
    e.set<std::array<unsigned short, 3>>("aStdArrUnsignedShort3", vStdArrUnsignedShort3);
    e.set<std::array<unsigned short, 4>>("aStdArrUnsignedShort4", vStdArrUnsignedShort4);
    e.set<std::size_t>("aStdSizeT", vStdSizeT);
    e.set<std::string>("aStdString", vStdString);
    e.set<std::vector<std::size_t>>("aVecStdSizeT", vVecStdSizeT);
    e.set<unsigned short>("aUnsignedShort", vUnsignedShort);

}

TEST(element, def_attrs)
{

    mv::Element e("TestElement");
    setAllAttrTypes(e);
    
    ASSERT_EQ(e.get<bool>("aBool"), vBool);
    ASSERT_EQ(e.get<double>("aDouble"), vDouble);
    ASSERT_EQ(e.get<mv::DType>("aDType"), vDType);
    ASSERT_EQ(e.get<int>("aInt"), vInt);
    ASSERT_EQ(e.get<mv::Order>("aOrder"), vOrder);
    ASSERT_EQ(e.get<mv::Shape>("aShape"), vShape);
    auto r1 = e.get<std::array<unsigned short, 2>>("aStdArrUnsignedShort2");
    ASSERT_EQ(r1, vStdArrUnsignedShort2);
    auto r2 = e.get<std::array<unsigned short, 3>>("aStdArrUnsignedShort3");
    ASSERT_EQ(r2, vStdArrUnsignedShort3);
    auto r3 = e.get<std::array<unsigned short, 4>>("aStdArrUnsignedShort4");
    ASSERT_EQ(r3, vStdArrUnsignedShort4);
    ASSERT_EQ(e.get<std::size_t>("aStdSizeT"), vStdSizeT);
    ASSERT_EQ(e.get<std::string>("aStdString"), vStdString);
    ASSERT_EQ(e.get<std::vector<std::size_t>>("aVecStdSizeT"), vVecStdSizeT);
    ASSERT_EQ(e.get<unsigned short>("aUnsignedShort"), vUnsignedShort);
    
}

TEST(element, get_failure)
{

    mv::Logger::instance().setVerboseLevel(mv::Logger::VerboseLevel::VerboseSilent);

    mv::Element e("TestElement");
    setAllAttrTypes(e);

    ASSERT_ANY_THROW(e.get<int>("aBool"));
    ASSERT_ANY_THROW(e.get<bool>("aDouble"));
    ASSERT_ANY_THROW(e.get<int>("aDType"));
    ASSERT_ANY_THROW(e.get<bool>("aInt"));
    ASSERT_ANY_THROW(e.get<unsigned short>("aOrder"));
    ASSERT_ANY_THROW(e.get<std::vector<std::size_t>>("aShape"));
    ASSERT_ANY_THROW(e.get<std::vector<std::size_t>>("aStdArrUnsignedShort2"));
    ASSERT_ANY_THROW(e.get<std::vector<std::size_t>>("aStdArrUnsignedShort3"));
    ASSERT_ANY_THROW(e.get<std::vector<std::size_t>>("aStdArrUnsignedShort4"));
    ASSERT_ANY_THROW(e.get<int>("aStdSizeT"));
    ASSERT_ANY_THROW(e.get<int>("aStdString"));
    ASSERT_ANY_THROW(e.get<mv::Shape>("aVecStdSizeT"));
    ASSERT_ANY_THROW(e.get<int>("aUnsignedShort"));

    mv::Logger::instance().setVerboseLevel(mv::Logger::VerboseLevel::VerboseError);

}

struct UnregisteredAttr
{

    int val;

};

TEST(element, set_unregistered)
{

    mv::Logger::instance().setVerboseLevel(mv::Logger::VerboseLevel::VerboseSilent);

    mv::Element e("TestElement");
    ASSERT_ANY_THROW(e.set<UnregisteredAttr>("a1", {0}));

    mv::Logger::instance().setVerboseLevel(mv::Logger::VerboseLevel::VerboseError);

}

TEST(element, get_unregisterd)
{

    mv::Logger::instance().setVerboseLevel(mv::Logger::VerboseLevel::VerboseSilent);

    mv::Element e("TestElement");
    setAllAttrTypes(e);

    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aBool"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aDouble"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aDType"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aInt"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aOrder"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aShape"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aStdArrUnsignedShort2"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aStdArrUnsignedShort3"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aStdArrUnsignedShort4"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aStdSizeT"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aStdString"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aVecStdSizeT"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aUnsignedShort"));

    mv::Logger::instance().setVerboseLevel(mv::Logger::VerboseLevel::VerboseError);

}

TEST(element, clear)
{

    mv::Element e("TestElement");
    setAllAttrTypes(e);
    e.clear();

    ASSERT_EQ(e.attrsCount(), 0);

}

TEST(element, to_json)
{

    mv::Element e("TestElement");
    setAllAttrTypes(e);

    std::string jsonStr = "{\"aBool\":{\"attrType\":\"bool\",\"content\":true},\"aDType\":{\"attrType\":\"DType\""
        ",\"content\":\"Float16\"},\"aDouble\":{\"attrType\":\"double\",\"content\":1.0},\"aInt\":{\"attrType\":\"int\""
        ",\"content\":2},\"aOrder\":{\"attrType\":\"Order\",\"content\":\"ColumnMajor\"},\"aShape\":{\"attrType\":\"Shape\""
        ",\"content\":[1,2,3]},\"aStdArrUnsignedShort2\":{\"attrType\":\"std::array<unsigned short COMMA 2>\",\"content\""
        ":[4,5]},\"aStdArrUnsignedShort3\":{\"attrType\":\"std::array<unsigned short COMMA 3>\",\"content\":[6,7,8]}"
        ",\"aStdArrUnsignedShort4\":{\"attrType\":\"std::array<unsigned short COMMA 4>\",\"content\":[9,10,11,12]}"
        ",\"aStdSizeT\":{\"attrType\":\"std::size_t\",\"content\":3},\"aStdString\":{\"attrType\":\"std::string\""
        ",\"content\":\"str\"},\"aUnsignedShort\":{\"attrType\":\"unsigned short\",\"content\":4},\"aVecStdSizeT\""
        ":{\"attrType\":\"std::vector<std::size_t>\",\"content\":[13,14,15,16,17]}}";

    ASSERT_EQ(e.toJSON().stringify(), jsonStr);

}