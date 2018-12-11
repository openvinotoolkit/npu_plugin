#include "include/mcm/utils/deployer/executor.hpp"
#include "include/mcm/utils/deployer/deployer_utils.hpp"

using namespace mv;
using namespace exe;

int main()
{
    Logger::setVerboseLevel(VerboseLevel::Info);
    std::string graphFile = utils::projectRootPath() + std::string("/tests/data/gold_11.blob");
    std::cout << "graph file " << graphFile << std::endl;
    Executor exec;
    Order order("NHWC");
    Shape shape({64, 64 ,3 ,1});

    Tensor inputTensor = mv::exe::dep_utils::getInputData(mv::exe::dep_utils::InputMode::ALL_ZERO, order, shape);
    Tensor res = exec.execute(graphFile, inputTensor);
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