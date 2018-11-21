#include "include/mcm/utils/hardware_tests.hpp"

mv::ReturnCodes mv::HWTest(mv::CompilationUnit& unit, std::string outputName, bool fathomHardware)
{
    auto compOutput = unit.run();

    mv::ReturnCodes toReturn;

    system(std::string("dot -Tsvg " + outputName + ".dot -o " + outputName + ".svg").c_str());
    system(std::string("dot -Tsvg " + outputName + "_adapt.dot -o " + outputName + "_adapt.svg").c_str());
    system(std::string("dot -Tsvg " + outputName + "_final.dot -o " + outputName + "_final.svg").c_str());

    //0) Run generated blob
    std::cout << "RUNNING ON HW MCMCOMPILER BLOB" << std::endl;
    std::string command0("python3 " + mv::utils::projectRootPath() + "/python/tools/mcmRunHW.py --blob " + outputName + ".blob --result " + outputName + ".npy --image " + mv::utils::projectRootPath() + "/mug.png");
    std::cout << command0 << std::endl;
    toReturn.mcmBlobOnHardware = system(command0.c_str());

    //1) Compile generated prototxt with fathom
    std::cout << "COMPILING GENERATED PROTOTXT WITH FATHOM" << std::endl;
    std::string command1("python3 " + mv::utils::mdkRootPath() + "/projects/Fathom/src2/mvNCCompile.py " + outputName + ".prototxt -w " + outputName + ".caffemodel ");
    if(fathomHardware)
        command1 += " --ma2480";
    std::cout << command1 << std::endl;
    toReturn.fathomCompilation = system(command1.c_str());

    //1b) Run diff on blobs
    std::cout << "DIFF COMMAND ON BLOBS" << std::endl;
    std::string diffCommand("diff " + outputName + ".blob graph");
    std::cout << diffCommand << std::endl;
    toReturn.diffOutput = system(diffCommand.c_str());

    //2) Compare python and caffe
    std::cout << "COMPARING FATHOM AND CAFFE" << std::endl;
    std::string command3("python3 " + mv::utils::mdkRootPath() + "/projects/Fathom/src2/mvNCCheck.py " + outputName + ".prototxt -w " + outputName + ".caffemodel ");
    std::cout << command3 << std::endl;
    toReturn.fathomVsCaffe = system(command3.c_str());

    //3) Compare python and cpp
    std::cout << "COMPARING FATHOM AND MCMCOMPILER BLOBS" << std::endl;
    std::string command2(mv::utils::projectRootPath() + "/python/tools/mcmCheck.sh -b " + outputName + ".blob -b graph -i " + mv::utils::projectRootPath() + "/mug.png");
    std::cout << command2 << std::endl;
    toReturn.fathomVsMcm = system(command2.c_str());

    return toReturn;
}

void mv::printReport(mv::ReturnCodes returnValue, std::ostream& out)
{
    out << "FINAL REPORT" << std::endl;
    out << "MCM RUN ON HW " << returnValue.mcmBlobOnHardware << std::endl;
    out << "FATHOM COMPILATION " << returnValue.fathomCompilation << std::endl;
    out << "BLOB DIFF " << returnValue.diffOutput << std::endl;
    out << "FATHOM VS CAFFE " << returnValue.fathomVsCaffe << std::endl;
    out << "FATHOM VS MCM " << returnValue.fathomVsMcm << std::endl;
}

