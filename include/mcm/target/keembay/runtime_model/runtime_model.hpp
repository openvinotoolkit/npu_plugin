#ifndef MV_RUNTIME_MODEL_
#define MV_RUNTIME_MODEL_

#include "meta/schema/graphfile/graphfile_generated.h"

namespace mv
{
    class RuntimeModel
    {
        private:
            MVCNN::GraphFileT graphFile_;
        public:
            RuntimeModel();
            ~RuntimeModel();

            void serialize(const std::string& path);
            char * serialize(int& bufferSize);
            void deserialize(const std::string& path);
            void deserialize(char * buffer, int length);
    };
}

#endif
