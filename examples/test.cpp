#include <iostream>
#include "include/mcm/base/attribute.hpp"
#include "include/mcm/base/element.hpp"
#include <vector>

int main()
{

    mv::Attribute v1;
    if (!v1.valid())
        std::cout << "NULL" << std::endl;

    mv::Attribute v2 = 9.0;
    std::cout << v2.getTypeName() << std::endl;
    std::cout << v2.get<double>() << std::endl;
    std::cout << v2.toString() << std::endl;
    std::cout << v2.toJSON().stringifyPretty() << std::endl;

    mv::Attribute v3 = 1.0;

    std::cout << v3.toString() << std::endl;

    mv::json::Value j1 = v3.toJSON();
    std::cout << j1.stringifyPretty() << std::endl;

    mv::Attribute v4 = j1;

    std::cout << v4.toString() << std::endl;
    std::cout << v4.toJSON().stringifyPretty() << std::endl;

    //mv::json::Value j2((long long)1);
    //mv::Attribute v5 = j2;

    mv::Element e1("e1");
    e1.set<double>("a1", 1.0);
    e1.set<double>("a2", 2.0);
    e1.set<std::string>("a3", "str1");
    e1.set<std::vector<std::size_t>>("a4", {1, 2 ,3});
    e1.set<int>("a5", -1);
    e1.set<std::size_t>("a6", 10);

    std::cout << e1.toString() << std::endl;
    std::cout << e1.toJSON().stringifyPretty() << std::endl;

    mv::json::Value j2 = e1.toJSON();
    mv::Element e2(j2);

    std::cout << e2.toString() << std::endl;
    std::cout << e2.get<double>("a1") << std::endl;
    std::cout << e2.get<std::string>("a3") << std::endl;
    std::cout << e2.get<std::size_t>("a6") << std::endl;

}