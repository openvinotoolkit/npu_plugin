//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "opManager.h"
#include "NoOp.h"
#include "CustomCpp.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

Op * opManager::createOp(t_MvTensorOpType which_one, int OpPosition, int /*numberOfNCEs*/) {

    if (OpPosition == primaryOperation) {
        switch(which_one) {
        case kCustomCpp:
            return new CustomCpp(which_one);
        default:
            printf("NO SUCH Op STAGE: %i\n", which_one);
            return new NoOp(kNone0);
        }
    } else {
        printf("No SUCH STAGE\n");
        return new NoOp(kNone0);
    }
}
