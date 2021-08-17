// {% copyright %}

#ifndef SHARED_MODULES_POOLING_H_
#define SHARED_MODULES_POOLING_H_

#include "Op.h"

typedef struct
{
    unsigned int excludePad;
} t_PoolingLayerParams;

class Pooling: public Op
{
public:
    Pooling() = default;
    Pooling(t_MvTensorOpType op_type) : Op(op_type) {
        radixX = 0;
        radixY = 0;
        radixStrideX = 0;
        radixStrideY = 0;
        padX = 0;
        padY = 0;
        ops.excludePad = 0;
    }
    virtual ~Pooling();

    virtual void run(mv::tensor::Processor& mvtp,
            t_MvTensorMyriadResources& myriadRes,
            t_MvTensorDebugInfo& debugInfo) override;

    int radixX;
    int radixY;
    int radixStrideX;
    int radixStrideY;
    int padX;
    int padY;

    int32_t inIndices[MAX_DIMS];
    int32_t outIndices[MAX_DIMS];

    t_PoolingLayerParams ops;

    Buffer input;
    Buffer output;
    std::string pool_method;
private:
    void calcPad(const Buffer& input, const Buffer& output, int& rpad_x, int& bpad_y);

    int batch_dim = 1;
    int input_batch_step = 1;
    int output_batch_step = 1;
};

#endif /* SHARED_MODULES_POOLING_H_ */
