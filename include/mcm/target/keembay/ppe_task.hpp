#ifndef MV_PPETASK_
#define MV_PPETASK_

#include <string>
#include <array>
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/target/keembay/ppe_fixed_function.hpp"

namespace mv
{
    class PPETask : public LogSender
    {
        private:
            std::shared_ptr<Tensor> scaleData_;
            PPEFixedFunction fixedFunction_;

        public:
            std::string toString() const;
            PPETask();
            PPETask(const std::string& value);
            ~PPETask();
            std::string getLogID() const;
    };
}

#endif
