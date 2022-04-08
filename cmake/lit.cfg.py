#
# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0
#

import lit.formats
import lit.llvm

config.test_format = lit.formats.ShTest()

lit.llvm.llvm_config.with_system_environment(['HOME', 'TMP', 'TEMP', 'OV_BUILD_DIR'])
lit.llvm.llvm_config.with_environment('PATH', config.bin_dir, append_path=True)

config.environment['FILECHECK_OPTS'] = '-enable-var-scope --allow-unused-prefixes=false --color -v'
