// {% copyright %}
#ifndef MV_TENSOR_CPP_H_
#define MV_TENSOR_CPP_H_

#include "mvTensor.h"
#include "mvTensorResources.h"

namespace mv
{
    namespace tensor
    {
        class Processor
        {
        public:
            Processor(
                const t_MvTensorMyriadResources &myriadResources,
                const t_MvTensorDebugInfo *debugInfo);

            const Resources& getResources() const { return resources_; }

        private:
            t_MvTensorMyriadResources myriadResources_;
            OutputStream debugStream_;
            Resources resources_;
            int prevShaves_;

            Processor(const Processor &);
            Processor &operator =(const Processor &);
        };
    }
}

#endif // MV_TENSOR_CPP_H_
