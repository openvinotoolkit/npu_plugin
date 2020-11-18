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

import lit.formats
import lit.llvm

config.test_format = lit.formats.ShTest()

lit.llvm.llvm_config.with_system_environment(['HOME', 'TMP', 'TEMP'])
lit.llvm.llvm_config.with_environment('PATH', config.bin_dir, append_path=True)

config.environment['FILECHECK_OPTS'] = '-enable-var-scope --allow-unused-prefixes=false --color -v'
