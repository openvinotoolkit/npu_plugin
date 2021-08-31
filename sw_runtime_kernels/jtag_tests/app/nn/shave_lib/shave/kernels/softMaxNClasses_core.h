// {% copyright %}

#ifndef _MVSOFTMAXNCLASSES_CORE_H_
#define _MVSOFTMAXNCLASSES_CORE_H_

#include <param_softmax.h>
#ifdef __cplusplus
extern "C"
{
#endif
using namespace nn;
using namespace shave_lib;

typedef enum
{
    working = 10,
    finished = 20,
    aggregated = 30,
    invalid = 30,
}execution_state;

typedef struct
{
    volatile execution_state status;
    volatile fp16 largest;
    volatile float sumf;
    volatile fp16 reciprocal_sum;
}intershaves;

void mvSoftMax  (t_MvSoftMaxParamNClasses* p);
void mvSoftMaxInner(t_MvSoftMaxParamNClasses* p);
void mvSoftMaxInner_1x1xN(t_MvSoftMaxParamNClasses* p);

#ifdef __cplusplus
}
#endif

#endif//_MVSOFTMAXNCLASSES_CORE_H_
