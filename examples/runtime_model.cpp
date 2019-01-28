#include "include/mcm/target/keembay/runtime_model/runtime_model.hpp"
#include <iostream>

int main()
{
    mv::RuntimeModel rm;
    rm.serialize("test.blob");
    mv::RuntimeModel rm2;
    rm2.deserialize("test.blob");
    return 0;
}
