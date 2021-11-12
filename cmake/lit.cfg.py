#
# Copyright Intel Corporation.
#
# LEGAL NOTICE: Your use of this software and any required dependent software
# (the "Software Package") is subject to the terms and conditions of
# the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
# which may also include notices, disclaimers, or license terms for
# third party or open source software included in or with the Software Package,
# and your use indicates your acceptance of all such terms. Please refer
# to the "third-party-programs.txt" or other similarly-named text file
# included with the Software Package for additional details.
#

import lit.formats
import lit.llvm

config.test_format = lit.formats.ShTest()

lit.llvm.llvm_config.with_system_environment(['HOME', 'TMP', 'TEMP', 'MV_TOOLS_DIR', 'MV_TOOLS_VERSION'])
lit.llvm.llvm_config.with_environment('PATH', config.bin_dir, append_path=True)

config.environment['FILECHECK_OPTS'] = '-enable-var-scope --allow-unused-prefixes=false --color -v'
