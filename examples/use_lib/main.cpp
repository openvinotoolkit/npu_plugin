#include <include/mcm/computation/model/op_model.hpp>
#include <iostream>

int main()
{

    mv::OpModel om("test");
    auto inIt = om.input({1, 32, 32, 3}, mv::DType("Float16"), mv::OrderType::ColumnMajor);
    auto outIt = om.output(inIt);
    
    if (om.isValid())
        std::cout << "Valid model" << std::endl;
    else
        std::cout << "Invalid model" << std::endl;
    
    return 0;

}