// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dims_tests.hpp"

#ifdef USE_MYRIAD
    PLUGING_CASE_WITH_SUFFIX(myriad, _nightly, IO_BlobTest, params_myriad);
#endif
#ifdef USE_HDDL
    PLUGING_CASE_WITH_SUFFIX(HDDL, _nightly, IO_BlobTest, params_myriad);
#endif
#ifdef USE_KMB
    PLUGING_CASE_WITH_SUFFIX(kmb, _nightly, IO_BlobTest, params_myriad);
#endif
