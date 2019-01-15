#ifndef MV_ENV_LOADER_HPP_
#define MV_ENV_LOADER_HPP_

#include <string>

namespace mv
{

    namespace utils
    {

        std::string projectRootPath();
        std::string mdkRootPath();
        bool fileExists(const std::string& fileName);

    }

}

#endif // MV_ENV_LOADER_HPP_
