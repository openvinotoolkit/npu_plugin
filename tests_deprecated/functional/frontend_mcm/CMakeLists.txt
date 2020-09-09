#
# Copyright 2020 Intel Corporation.
#
# This software and the related documents are Intel copyrighted materials,
# and your use of them is governed by the express license under which they
# were provided to you (End User License Agreement for the Intel(R) Software
# Development Products (Version May 2017)). Unless the License provides
# otherwise, you may not use, modify, copy, publish, distribute, disclose or
# transmit this software or the related documents without Intel's prior
# written permission.
#
# This software and the related documents are provided as is, with no
# express or implied warranties, other than those that are expressly
# stated in the License.
#

set(TARGET_NAME "FrontendMCMFunctionalTests")

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT
            ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES
            "../../common_helpers"
            ${CMAKE_CURRENT_SOURCE_DIR}
        LINK_LIBRARIES
            IE::inference_engine
            IE::IESharedTests
            frontend_mcm
        LINK_LIBRARIES_WHOLE_ARCHIVE
            kmbTestData
        LABELS  # Must be the last parameter. Reason: see description comment for addIeTargetTest().
            KMB
            VPU
)

enable_warnings_as_errors(${TARGET_NAME})
