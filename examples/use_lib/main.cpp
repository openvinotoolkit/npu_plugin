#include <include/fathom/computation/model/op_model.hpp>
#include <iostream>

int main()
{

    mv::OpModel om(mv::Logger::VerboseLevel::VerboseInfo);
    auto inIt = om.input(mv::Shape(1, 32, 32, 3), mv::DType::Float, mv::Order::NWHC);
    auto outIt = om.output(inIt);
    
    if (om.isValid())
        std::cout << "Valid model" << std::endl;
    else
        std::cout << "Invalid model" << std::endl;
    
    return 0;

}