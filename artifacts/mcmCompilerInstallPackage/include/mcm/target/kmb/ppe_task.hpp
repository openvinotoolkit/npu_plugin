#ifndef MV_PPETASK_
#define MV_PPETASK_

#include <string>
#include <array>
#include "include/mcm/tensor/tensor.hpp"
#include "include/mcm/computation/model/iterator/tensor.hpp"
#include "include/mcm/target/kmb/ppe_fixed_function.hpp"

namespace mv
{
    class PPETask : public Element
    {
        public:
            PPETask(const json::Value& content);
            PPETask(const PPEFixedFunction &fixedFunction);

            inline mv::Data::TensorIterator getScaleData() const
            {
                return get<Data::TensorIterator>("scaleData");
            }

            inline PPEFixedFunction getFixedFunction() const
            {
                return get<PPEFixedFunction>("fixedFunction");
            }

            virtual std::string getLogID() const override;
            virtual std::string toString() const override;
    };
}

#endif
