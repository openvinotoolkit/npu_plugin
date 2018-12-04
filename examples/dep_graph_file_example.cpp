#include "include/mcm/deployer/executor/configuration.hpp"
#include "include/mcm/deployer/executor/executor.hpp"

int main()
{
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);
    //Create Configuration
    std::string graphFile(std::getenv("MDK_HOME"));
    graphFile += "/projects/Fathom/ncsdk/examples/caffe/GoogLeNet/graph";
    mv::Configuration config(graphFile);
    std::cout << "Configuration graph file " << config.getGraphFilePath() << std::endl;
    mv::Executor exec(config);
    mv::Tensor res = exec.execute();
    std::cout << "res Order " << res.getOrder().toString() << std::endl;
    std::cout << "res Shape " << res.getShape().toString() << std::endl;
    std::cout << "ndims " << res.getShape().ndims() << std::endl;
    std::cout << "totalSize " << res.getShape().totalSize() << std::endl;
    for (unsigned int i=0; i < res.getShape().totalSize(); i++)
        if (res(i) != 0)
            std::cout << "res[" << i << "] = " << res(i) << std::endl;
}