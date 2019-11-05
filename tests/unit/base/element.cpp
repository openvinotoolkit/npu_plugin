// TODO Handle all cases

#include "gtest/gtest.h"
#include "include/mcm/base/element.hpp"
#include "include/mcm/tensor/dtype/dtype.hpp"
#include "include/mcm/tensor/order/order.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/op_model.hpp"
#include "include/mcm/computation/model/control_model.hpp"
#include "include/mcm/computation/model/data_model.hpp"

static bool vBool = true;
static double vDouble = 1.0;
static mv::DType vDType(mv::DType("Float16"));
static int vInt = 2;
static mv::Order vOrder(mv::Order("CHW"));
static mv::Shape vShape({1, 2, 3});
static std::array<unsigned short, 2> vStdArrUnsignedShort2 = {4, 5};
static std::array<unsigned short, 3> vStdArrUnsignedShort3 = {6, 7, 8};
static std::array<unsigned short, 4> vStdArrUnsignedShort4 = {9, 10, 11, 12};
static std::size_t vStdSizeT = 3;
static std::string vStdString = "str";
static std::vector<std::size_t> vVecStdSizeT({13, 14, 15, 16, 17});
static std::vector<std::string> vVecStdString({"e1", "e2", "e3", "e4", "e5"});
static unsigned short vUnsignedShort = 4;

static std::string aBoolName = "aBool";
static std::string aDoubleName = "aDouble";
static std::string aDTypeName = "aDType";
static std::string aIntName = "aInt";
static std::string aOrderName = "aOrder";
static std::string aShapeName = "aShape";
static std::string aStdArrUnsignedShort2Name = "aStdArrUnsignedShort2";
static std::string aStdArrUnsignedShort3Name = "aStdArrUnsignedShort3";
static std::string aStdArrUnsignedShort4Name = "aStdArrUnsignedShort4";
static std::string aStdStringName = "aStdString";
static std::string aStdSizeTName = "aStdSizeT";
static std::string aVecStdSizeTName = "aVecStdSizeT";
static std::string aVecStdStringName = "aVecStdString";
static std::string aUnsignedShortName = "aUnsignedShort";

static std::string dataOpListIteratorName = "aDataOpListIteratorName";
static std::string dataTensorIteratorName = "aDataTensorIteratorName";

static void setValueAttrTypes(mv::Element& e)
{

    e.clear();
    e.set<bool>(aBoolName, vBool);
    e.set<double>(aDoubleName, vDouble);
    e.set<mv::DType>(aDTypeName, vDType);
    e.set<int>(aIntName, vInt);
    e.set<mv::Order>(aOrderName, vOrder);
    e.set<mv::Shape>(aShapeName, vShape);
    e.set<std::array<unsigned short, 2>>(aStdArrUnsignedShort2Name, vStdArrUnsignedShort2);
    e.set<std::array<unsigned short, 3>>(aStdArrUnsignedShort3Name, vStdArrUnsignedShort3);
    e.set<std::array<unsigned short, 4>>(aStdArrUnsignedShort4Name, vStdArrUnsignedShort4);
    e.set<std::size_t>(aStdSizeTName, vStdSizeT);
    e.set<std::string>(aStdStringName, vStdString);
    e.set<std::vector<std::size_t>>(aVecStdSizeTName, vVecStdSizeT);
    e.set<unsigned short>(aUnsignedShortName, vUnsignedShort);
    e.set<std::vector<std::string>>(aVecStdStringName, vVecStdString);

}

/*static void setPointerAttrTypes(mv::Element& e, mv::OpModel& m)
{
    e.clear();
    m.clear();

    auto input = m.input(mv::Shape({32, 32, 3}), mv::DType("Float16"), mv::Order("CHW"));
    m.output(input);
    auto inputOp = m.getSourceOp(input);

    e.set<mv::Data::OpListIterator>(dataOpListIteratorName, inputOp);
    e.set<mv::Data::TensorIterator>(dataTensorIteratorName, input);

}*/

TEST(element, def_attrs)
{

    mv::Element e("TestElement");
    setValueAttrTypes(e);

    ASSERT_TRUE(e.hasAttr(aBoolName));
    ASSERT_TRUE(e.hasAttr(aDoubleName));
    ASSERT_TRUE(e.hasAttr(aDTypeName));
    ASSERT_TRUE(e.hasAttr(aIntName));
    ASSERT_TRUE(e.hasAttr(aOrderName));
    ASSERT_TRUE(e.hasAttr(aShapeName));
    ASSERT_TRUE(e.hasAttr(aStdArrUnsignedShort2Name));
    ASSERT_TRUE(e.hasAttr(aStdArrUnsignedShort3Name));
    ASSERT_TRUE(e.hasAttr(aStdArrUnsignedShort4Name));
    ASSERT_TRUE(e.hasAttr(aStdStringName));
    ASSERT_TRUE(e.hasAttr(aStdSizeTName));
    ASSERT_TRUE(e.hasAttr(aVecStdSizeTName));
    ASSERT_TRUE(e.hasAttr(aUnsignedShortName));

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
    ASSERT_EQ(e.get<std::vector<std::string>>("aVecStdString"), vVecStdString);

}

TEST(element, get_failure)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    mv::Element e("TestElement");
    setValueAttrTypes(e);

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

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

struct UnregisteredAttr
{

    int val;

};

TEST(element, set_unregistered)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    mv::Element e("TestElement");
    ASSERT_ANY_THROW(e.set<UnregisteredAttr>("a1", {0}));

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

TEST(element, get_unregisterd)
{

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Silent);

    mv::Element e("TestElement");
    setValueAttrTypes(e);

    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aBool"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aDouble"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aDType"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aInt"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("amv::Order"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aShape"));

    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aStdArrUnsignedShort2"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aStdArrUnsignedShort3"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aStdArrUnsignedShort4"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aStdSizeT"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aStdString"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aVecStdSizeT"));
    ASSERT_ANY_THROW(e.get<UnregisteredAttr>("aUnsignedShort"));

    mv::Logger::instance().setVerboseLevel(mv::VerboseLevel::Error);

}

TEST(element, clear)
{

    mv::Element e("TestElement");
    setValueAttrTypes(e);
    e.clear();

    ASSERT_EQ(e.attrsCount(), 0);
    ASSERT_FALSE(e.hasAttr(aBoolName));
    ASSERT_FALSE(e.hasAttr(aDoubleName));
    ASSERT_FALSE(e.hasAttr(aDTypeName));
    ASSERT_FALSE(e.hasAttr(aIntName));
    ASSERT_FALSE(e.hasAttr(aOrderName));
    ASSERT_FALSE(e.hasAttr(aShapeName));
    ASSERT_FALSE(e.hasAttr(aStdArrUnsignedShort2Name));
    ASSERT_FALSE(e.hasAttr(aStdArrUnsignedShort3Name));
    ASSERT_FALSE(e.hasAttr(aStdArrUnsignedShort4Name));
    ASSERT_FALSE(e.hasAttr(aStdStringName));
    ASSERT_FALSE(e.hasAttr(aStdSizeTName));
    ASSERT_FALSE(e.hasAttr(aVecStdSizeTName));
    ASSERT_FALSE(e.hasAttr(aUnsignedShortName));

}

TEST(element, erase)
{

    mv::Element e("TestElement");
    setValueAttrTypes(e);
    std::size_t s = e.attrsCount();
    e.erase(aBoolName);
    ASSERT_FALSE(e.hasAttr(aBoolName));
    ASSERT_EQ(e.attrsCount(), s - 1);

}

TEST(element, to_json)
{

    mv::Element e("TestElement");
    setValueAttrTypes(e);

    std::string jsonStr =
        "{\"attrs\":{\"aBool\":{\"attrType\":\"bool\",\"content\":true},\"aDType\":"
        "{\"attrType\":\"DType\",\"content\":\"Float16\"},\"aDouble\":{\"attrType\""
        ":\"double\",\"content\":1.0},\"aInt\":{\"attrType\":\"int\",\"content\":2}"
        ",\"aOrder\":{\"attrType\":\"Order\",\"content\":\"CHW\"}"
        ",\"aShape\":{\"attrType\":\"Shape\",\"content\":[1,2,3]},\"aStdArrUnsignedShort2\":{\""
        "attrType\":\"std::array<unsigned short, 2>\",\"content\":[4,5]},\"aStdArrUn"
        "signedShort3\":{\"attrType\":\"std::array<unsigned short, 3>\",\"content\":"
        "[6,7,8]},\"aStdArrUnsignedShort4\":{\"attrType\":\"std::array<unsigned shor"
        "t, 4>\",\"content\":[9,10,11,12]},\"aStdSizeT\":{\"attrType\":\"std::size_t"
        "\",\"content\":3},\"aStdString\":{\"attrType\":\"std::string\",\"content\":"
        "\"str\"},\"aUnsignedShort\":{\"attrType\":\"unsigned short\",\"content\":4}"
        ",\"aVecStdSizeT\":{\"attrType\":\"std::vector<std::size_t>\",\"content\":[1"
        "3,14,15,16,17]},\"aVecStdString\":{\"attrType\":\"std::vector<std::string>"
        "\",\"content\":[\"e1\",\"e2\",\"e3\",\"e4\",\"e5\"]}},"
        "\"name\":\"TestElement\"}";

    ASSERT_EQ(e.toJSON().stringify(), jsonStr);
}

/*TEST(element, def_op_iterator_attr)
{

    mv::Element e("TestElement");
    mv::OpModel m("TestModel");
    setPointerAttrTypes(e, m);

    ASSERT_TRUE(e.hasAttr(dataOpListIteratorName));
    ASSERT_TRUE(e.hasAttr(dataTensorIteratorName));
    ASSERT_EQ(e.get<mv::Data::OpListIterator>(dataOpListIteratorName), m.getInput());
    ASSERT_EQ(e.get<mv::Data::TensorIterator>(dataTensorIteratorName), m.getInput()->getOutputTensor(0));

}*/

TEST(element, attribute_type_std_map) {

    mv::Element e("MapTestElement");
    mv::Element me1("map_elem1");
    mv::Element me2("map_elem2");

    static std::vector<std::string> vVecStdString1({"foo1", "foo2", "foo3", "foo4", "foo5"});
    static std::vector<std::string> vVecStdString2({"bar1", "bar2", "bar3", "bar4", "bar5"});
    static std::string aVecStdStringName1 = "aVecStdString1";
    static std::string aVecStdStringName2 = "aVecStdString2";

    me1.set<std::vector<std::string>>(aVecStdStringName1, vVecStdString1);
    me2.set<std::vector<std::string>>(aVecStdStringName2, vVecStdString2);

    static std::map<std::string, mv::Element>  e_map;
    e_map.emplace("key1", me1);
    e_map.emplace("key2", me2);

    e.set<std::map<std::string, mv::Element>>("map_attribute", e_map);
    //std::cout << e.toString() << std::endl;

    std::string jsonStr = "{\"attrs\":{\"map_attribute\":{\"attrType\":\"std::map"
    "<std::string, Element>\",\"content\":{\"key1\":{\"attrs\":{\"aVecStdString1\":"
    "{\"attrType\":\"std::vector<std::string>\",\"content\":[\"foo1\",\"foo2\",\"fo"
    "o3\",\"foo4\",\"foo5\"]}},\"name\":\"map_elem1\"},\"key2\":{\"attrs\":{\"aVecS"
    "tdString2\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"bar1\",\"b"
    "ar2\",\"bar3\",\"bar4\",\"bar5\"]}},\"name\":\"map_elem2\"}}}},\"name\":\"MapTe"
    "stElement\"}";

    ASSERT_EQ(e.toJSON().stringify(), jsonStr);

}

TEST(element, element_from_json) {
    mv::Element e1("groups");

    e1.set("dVal", 42.0);
    e1.set("sVal", std::string("a_string"));
    e1.set("iVal", 84);
    e1.set<mv::Element>("group1", mv::Element("group1"));
    e1.set<std::vector<std::string>>("strVec", {});
    e1.get<std::vector<std::string>>("strVec").push_back("v1");
    e1.get<std::vector<std::string>>("strVec").push_back("v2");
    e1.get<std::vector<std::string>>("strVec").push_back("v3");
    e1.get<mv::Element>("group1").set<std::vector<double>>("doubleVec1", {1.0, 2.0, 3.0});
    e1.get<mv::Element>("group1").set<std::vector<double>>("doubleVec2", {4.0, 5.0, 6.0});
    e1.get<mv::Element>("group1").set<mv::Element>("group2", mv::Element("group2"));
    e1.get<mv::Element>("group1").get<mv::Element>("group2").set("dVal2", 42.0);
    e1.get<mv::Element>("group1").get<mv::Element>("group2").set<std::vector<std::string>>("strVec2", {"val1", "val2", "val3"});

    mv::json::Value obj = e1.toJSON(true);

    mv::Element e2(obj, true);

    mv::json::Value obj2 = e2.toJSON(true);

    ASSERT_EQ(e1, e2);
}

TEST(element, attribute_type_std_vector_element) {

    mv::Element e("testVecElementAttribute");
    mv::Element me1("vec_elem1");
    mv::Element me2("vec_elem2");

    std::vector<std::string> vVecStdString1({"foo1", "foo2", "foo3", "foo4", "foo5"});
    std::vector<std::string> vVecStdString2({"bar1", "bar2", "bar3", "bar4", "bar5"});
    std::string aVecStdStringName1 = "aVecStdString1";
    std::string aVecStdStringName2 = "aVecStdString2";

    me1.set<std::vector<std::string>>(aVecStdStringName1, vVecStdString1);
    me2.set<std::vector<std::string>>(aVecStdStringName2, vVecStdString2);

    std::vector<mv::Element>  elemVec;
    elemVec.push_back(me1);
    elemVec.push_back(me2);

    e.set<std::vector<mv::Element>>("elemVecAttr", elemVec);
    std::cout << e.toString() << std::endl;

    // std::string jsonStr = "{\"attrs\":{\"map_attribute\":{\"attrType\":\"std::map"
    // "<std::string, Element>\",\"content\":{\"key1\":{\"attrs\":{\"aVecStdString1\":"
    // "{\"attrType\":\"std::vector<std::string>\",\"content\":[\"foo1\",\"foo2\",\"fo"
    // "o3\",\"foo4\",\"foo5\"]}},\"name\":\"map_elem1\"},\"key2\":{\"attrs\":{\"aVecS"
    // "tdString2\":{\"attrType\":\"std::vector<std::string>\",\"content\":[\"bar1\",\"b"
    // "ar2\",\"bar3\",\"bar4\",\"bar5\"]}},\"name\":\"map_elem2\"}}}},\"name\":\"MapTe"
    // "stElement\"}";

    //ASSERT_EQ(e.toJSON().stringify(), jsonStr);

    std::cout << e.toJSON(true).stringifyPretty() << std::endl;

}