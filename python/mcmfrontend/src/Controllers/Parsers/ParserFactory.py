# Copyright 2018 Intel Corporation.
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

from Models.EnumDeclarations import Parser


from Controllers.Parsers.TensorFlowLite import TensorFlowLiteParser
print("TensorFlow Lite found")

from Controllers.Parsers.BaseParser import BaseParser


class ParserFactory(object):

    @staticmethod
    def getInstance(parser_enum: Parser) -> BaseParser:
        if parser_enum == Parser.TensorFlowLite:
            parser = TensorFlowLiteParser()
        else:
            parser = BaseParser()
        return parser
