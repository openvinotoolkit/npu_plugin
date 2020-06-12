#include "include/mcm/compiler/compilation_unit.hpp"
#include <iostream>
#include <fstream>

int main(int argc, char **argv)
{

    mv::CompilationUnit unit("weightSparsity");
    mv::OpModel& om = unit.model();

    auto input0 = om.input({64,64,64,1}, mv::DType("UInt8"), mv::Order::getZMajorID(4),  {{0},{0.00392156862745098},{},{}}, "input#170");

    auto pool0 = om.maxPool(input0, {1,1}, {1,1}, {0,0,0,0}, true, mv::DType("UInt8"), {{0},{0.00392156862745098},{},{}}, "pool");

    //NOTE: The idea here is that the first 64 elements of weights will be 1 and all the rest 0s. Doing that the whole output
    //tensor is expected to be full of 1s. Enable weight sparsity and validate...
    //In SOK, if the last cluster for example was taking full weight tensor of zeros, runtime would hang, runtime needs
    //to handle that case if it hits us in the future...
    std::vector<int64_t> weightsData0 = mv::utils::generateSequence<int64_t> (64, 255, 0);
    std::vector<int64_t> weightsData1 = mv::utils::generateSequence<int64_t> (4032, 0, 0);
    weightsData0.insert(weightsData0.end(), weightsData1.begin(), weightsData1.end());

    auto weights0 = om.constantInt(weightsData0,{1,1,64,64}, mv::DType("UInt8"), mv::Order::getRowMajorID(4), {{0},{0.00392156862745098},{},{}});
    auto conv0 = om.conv(pool0, weights0, {1, 1}, {0, 0, 0, 0}, 1, 1,  mv::DType("UInt8"),{{0},{0.00392156862745098},{},{}} , "conv");
    om.output(conv0);

    std::string compDescPath;
    if (argc == 2) {
      FILE *fptr = fopen(argv[1], "r");
      if (!fptr) {
        fprintf(stderr, "Cannot open the CD file %s\n", argv[1]);
      }
      compDescPath = argv[1];
    } else {
      printf("[Using the default CD]\n");
      compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    }
    unit.loadCompilationDescriptor(compDescPath);
    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
