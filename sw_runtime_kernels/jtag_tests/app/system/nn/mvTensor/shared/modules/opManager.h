// {% copyright %}
/*
 * opManager.h
 *
 *  Created on: Jun 1, 2017
 *      Author: ian-movidius
 */

#ifndef SHARED_MODULES_OPMANAGER_H_
#define SHARED_MODULES_OPMANAGER_H_

#include "Op.h"

class opManager
{
public:
    const uint32_t STAGE_BORDER_SYMBOL = 0x7f83ff19; // it is one of float NaN and prime number

    enum
    {
        primaryOperation = 0,
        preOperation = 1,
        postOperation = 2
    };

    opManager() = default;
    virtual ~opManager() = default;
//    Op * parseStage(const unsigned char * stageInBlob, unsigned &stageLenght, int numberOfNCEs = 1);

    Op * createOp(t_MvTensorOpType which_one, int OpPosition, int numberOfNCEs = 1);
};

#endif /* SHARED_MODULES_OPMANAGER_H_ */
