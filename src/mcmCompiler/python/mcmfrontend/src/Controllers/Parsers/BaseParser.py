# Copyright (C) 2022 Intel Corporation.
# SPDX-License-Identifier: Apache 2.0

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
