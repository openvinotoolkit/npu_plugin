#include "include/mcm/utils/hardware_tests.hpp"

void mv::HWTest(mv::CompilationUnit& unit, std::string outputName)
{
    auto compOutput = unit.run();

    system(std::string("dot -Tsvg " + outputName + ".dot -o " + outputName + ".svg").c_str());
    system(std::string("dot -Tsvg " + outputName + "_adapt.dot -o " + outputName + "_adapt.svg").c_str());
    system(std::string("dot -Tsvg " + outputName + "_final.dot -o " + outputName + "_final.svg").c_str());

    //0) Run generated blob
    std::cout << "RUNNING ON HW MCMCOMPILER BLOB" << std::endl;
    std::string command0("python3 " + mv::utils::projectRootPath() + "/python/tools/mcmRunHW.py --blob " + outputName + ".blob --result " + outputName + ".npy --image " + mv::utils::projectRootPath() + "/mug.png");
    std::cout << command0 << std::endl;
    system(command0.c_str());

    //1) Compile generated prototxt with fathom
    std::cout << "COMPILING GENERATED PROTOTXT WITH FATHOM" << std::endl;
    std::string command1("python3 " + mv::utils::mdkRootPath() + "/projects/Fathom/src2/mvNCCompile.py " + outputName + ".prototxt -w " + outputName + ".caffemodel ");
    std::cout << command1 << std::endl;
    system(command1.c_str());

    //2) Compare python and caffe
    std::cout << "COMPARING PYTHON AND CAFFE" << std::endl;
    std::string command3("python3 " + mv::utils::mdkRootPath() + "/projects/Fathom/src2/mvNCCheck.py " + outputName + ".prototxt -w " + outputName + ".caffemodel ");
    std::cout << command3 << std::endl;
    system(command3.c_str());

    //3) Compare python and cpp
    std::cout << "COMPARING FATHOM AND MCMCOMPILER BLOBS" << std::endl;
    std::string command2(mv::utils::projectRootPath() + "/python/tools/mcmCheck.sh -b " + outputName + ".blob -b graph -i " + mv::utils::projectRootPath() + "/mug.png");
    std::cout << command2 << std::endl;
    system(command2.c_str());


}
