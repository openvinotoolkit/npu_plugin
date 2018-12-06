#include "include/mcm/utils/deployer/configuration.hpp"
#include "include/mcm/utils/deployer/executor.hpp"

int main()
{
    mv::Logger::setVerboseLevel(mv::VerboseLevel::Info);
    //Create Configuration
    std::string graphFile = mv::utils::projectRootPath() + std::string("/tests/data/gold_11.blob");
    mv::Configuration config(graphFile);
    std::cout << "Configuration graph file " << config.getGraphFilePath() << std::endl;
    mv::Executor exec(config);
    mv::Tensor res = exec.execute();
    std::cout << "res Order " << res.getOrder().toString() << std::endl;
    std::cout << "res Shape " << res.getShape().toString() << std::endl;
    std::cout << "ndims " << res.getShape().ndims() << std::endl;
    std::cout << "totalSize " << res.getShape().totalSize() << std::endl;
    unsigned short max = 0;
    unsigned int max_idx = 0;
    for (unsigned int i=0; i < res.getShape().totalSize(); i++)
    {
        if (res(i) > max)
        {
            max = res(i);
            max_idx = i;
        }
        //if (res(i) != 0)
        //    std::cout << "res[" << i << "] = " << res(i) << std::endl;

    }
    std::cout << "res max idx " << max_idx << " val " << max << std::endl;
}