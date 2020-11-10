# Copyright 2019 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

from Controllers.EnumController import throw_error, ErrorTable


class BaseParser(object):

    def __init__(self):
        self.type = 'BaseParser'

    def getType(self):
        return self.type

    def calculateReference(self, arguments):
        throw_error(ErrorTable.ParserNotSupported,
                    "Abstract class BaseParser Instantiated")

    def loadNetworkObjects(self, graph_path, model_path=None):
        throw_error(ErrorTable.ParserNotSupported,
                    "Abstract class BaseParser Instantiated")

    def parse(self, arguments):
        throw_error(ErrorTable.ParserNotSupported,
                    "Abstract class BaseParser Instantiated")

    def get_layer_data(self, outBlobName):
        throw_error(ErrorTable.ParserNotSupported,
                    "Abstract class BaseParser Instantiated")
