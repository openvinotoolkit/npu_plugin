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

import numpy as np
from Controllers.Parsers.Parser.Layer import MangledName, OriginalName


class HwBarrier():
    def __init__(self, producers = [], consumers = []):
        self.setProducers(producers)
        self.setConsumers(consumers)

    def setProducers(self, producers):
        self.__producers = producers

    def getProducers(self):
        return self.__producers

    def addProducers(self, producers):
        self.__producers.append(producers)
    
    def setConsumers(self, consumers):
        self.__consumers = consumers
        self.name = MangledName(OriginalName("_".join([name for name in consumers])))

    def getConsumers(self):
        return self.__consumers

    def addConsumers(self, consumers):
        self.__consumers.append(consumers)

    def clean(self):
        self.__producers = []
        self.__consumers = []

    def isSet(self):
        return self.hasProducers() and self.hasConsumers()

    def hasProducers(self):
        return self.__producers != []

    def hasConsumers(self):
        return self.__consumers != []

    def setGroup(self, group):
        self.group = group

    def getGroup(self):
        return self.group

    def setIndex(self, index):
        self.index = index

    def getIndex(self):
        return self.index

    def __repr__(self):
        return "_".join(self.getConsumers())

