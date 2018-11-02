#include "include/mcm/computation/op/op_registry.hpp"

int main()
{
    mv::op::OpRegistry::generateCompositionAPI();
    mv::op::OpRegistry::generateRecordedCompositionAPI();
    return 0;
}