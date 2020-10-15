#include "include/mcm/computation/op/op_registry.hpp"

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: gen_composition_api <path to meta directory>" << std::endl;
        return 1;
    }

    mv::op::OpRegistry::generateCompositionAPI(argv[1]);

    return 0;
}
