#!/usr/bin/env python3

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


import networkx as nx

from collections import OrderedDict
from Controllers.GraphUtils import buildGraph, buildLayerLists
from Controllers.Tensor import Tensor, UnpopulatedTensor
from Controllers.Parsers.Parser.Conversion import Conversion, getDefaultConversionLayerFormatPool


def matchConsumingFormat(formatPool, matchingFormat, shape, tensorAxisDemand):
    def compatibleMatch(formatPair):
        consumingFmt, _ = formatPair

        return consumingFmt.compatible(matchingFormat, shape, tensorAxisDemand)

    return list(filter(compatibleMatch, formatPool))


def tensorChoiceOrPool(layer, tensor):
    if tensor.format:
        return tensor.format
    else:
        return layer.formatPool


def reducedFormatPool(producerMode, currentLayer):
    # If the inputTensor for a layer is set, we need to readjust the formatPool
    def adjustmentCheck(a):
        cons, prod = a
        pCons, pProd = producerMode
        # print('Filter', cons.layout, pProd.layout)
        return cons.layout == pProd.layout

    res = list(filter(adjustmentCheck, currentLayer.formatPool))
    return res


def findSolution(producerMode, consumer, tensor, ignoreEnclosure=False):
    """
        Given a producer format, search though the consumer formats
        to find a **single** best consumer format. Notice that priority
        layed out in formatPool matters here.

        Returns `None` if no solution exists.
    """

    solutionExists = None

    if ignoreEnclosure:
        tensorAxisDemand = [0, 0, 0, 0]
    else:
        tshape = tensor.getShape()
        eshape = tensor.getTopEncloserRecursive().getShape()
        tensorAxisDemand = [
            1 if x -
            y != 0 else 0 for (
                x,
                y) in zip(
                tshape,
                eshape)]

    for consumerMode in reducedFormatPool(producerMode, consumer):
        producerInputFmt, producerOutputFmt = producerMode
        consumerInputFmt, consumerOutputFmt = consumerMode

        solutionExists = producerOutputFmt.compatible(
            consumerInputFmt, tensor.getShape(), tensorAxisDemand)

        if solutionExists:
            # print('Found solution:', consumer.getStringifiedName(), producerOutputFmt.getSignature(), ' | ', consumerInputFmt.getSignature())
            # print(tensor.getShape())
            return (producerOutputFmt, consumerInputFmt)

    return solutionExists


def getVotesRaw(
        producerFormatPool,
        consumers,
        tensor,
        skipMarkedConsumers=False,
        ignoreEnclosure=False):
    """
        A producer provides the modes it can operate (`availableModes`) and the consumers
        answer whether they can correspond to these modes.
    """
    votes = OrderedDict()

    for pModeIdx, pMode in enumerate(producerFormatPool):
        for consumerIdx, consumer in enumerate(consumers):
            if skipMarkedConsumers and consumer == -1:
                continue

            sol = findSolution(
                pMode,
                consumer,
                tensor,
                ignoreEnclosure=ignoreEnclosure)
            if sol:
                try:
                    votes[pModeIdx].append((consumerIdx, sol))
                except BaseException:
                    votes[pModeIdx] = [(consumerIdx, sol)]

    return votes


def getVotes(producer, consumers, tensor, skipMarkedConsumers=False):
    return getVotesRaw(
        producer.formatPool,
        consumers,
        tensor,
        skipMarkedConsumers)


def minimizeConversionLayers(votes):
    if not votes:
        return []

    maxNum = 0
    maxKey = None
    for mode, capableConsumers in votes.items():
        if len(capableConsumers) > maxNum:
            maxNum = len(capableConsumers)
            maxKey = mode

    # print()
    # print('/////// Votes', votes)
    # print('/////// {} consumers agree on mode {}'.format(votes[maxKey], maxKey))

    satisfiedConsumers = votes[maxKey]
    del votes[maxKey]

    # Delete the satisfied consumers from everywhere
    for mode, capableConsumers in votes.items():
        l = capableConsumers

        def stripConsumerList(cList):
            return[c[0] for c in cList]

        def getConsumerId(votesEntry):
            return votesEntry[0]

        for i in stripConsumerList(satisfiedConsumers):
            l = list(filter(lambda x: getConsumerId(x) != i, l))

        votes[mode] = l

    # Delete empty modes
    for mode in list(votes.keys()):
        if not votes[mode]:
            del votes[mode]

    return [(maxKey, satisfiedConsumers)] + minimizeConversionLayers(votes)


def getPrevelantChoice(votes):
    bestMode = None
    bestLayers = []
    maxLayers = 0
    for mode, layers in minimizeConversionLayers(OrderedDict(votes)):
        if len(layers) > maxLayers:
            maxLayers = len(layers)
            bestMode = mode
            bestLayers = layers

    return (bestMode, bestLayers)


def fixTensors(parsedLayers, scheduler, myriadX=False):

    # Set the tensor formats for each layer
    # Notice that hardware layers only output in interleaved mode
    # (simplification)
    """
        If HwConv or HwFC, set the format for scheduling
    """
    for layer in parsedLayers:
        if not hasattr(layer, 'formatPool'):
            # If there is an error here, a software layer may not have had its
            # formatPool set.
            raise Exception(
                'Layer {} does not specify supported formats'.format(layer))

    # Set each tensor format to None, meaning that this tensor
    # has not been set to a particular format.
    """
        Initialize .format of all tensors to None
    """
    for layer in parsedLayers:
        for t in layer.getInputTensors():
            t.format = None
        for t in layer.getOutputTensors():
            t.format = None

    # Expand the graph, by showing both operations and tensors
    g = buildGraph(parsedLayers, contract_blobs=False)

    # Optimistically set the layout of tensors where consumer ops need only this
    # tensor and the producer together with all the consumers reach a consensus
    """
        For all Tensors that are individual inputs to Operations, vote on formats for the sequence.
    """
    for name in nx.lexicographical_topological_sort(g):
        layer_type = g.node[name]['type']

        if layer_type == 'BLOB':
            tensor = g.node[name]['ref']

            producerNode = list(g.predecessors(name))

            assert(len(producerNode) == 1)


            producer = g.node[producerNode[0]]['ref']

            consumersNodes = list(g.successors(name))
            consumers = [g.node[name]['ref'] for name in consumersNodes]

            # If there is a consumer that takes as input more than one input tensor, skip it.
            # Except if this consumer is a PriorBox
            skipTensor = False
            for consumer in consumers:
                if len(consumer.getInputTensors()) > 1:
                    skipTensor = True
                    break
            if skipTensor:
                continue

            votes = getVotes(producer, consumers, tensor)
            for producerFormatPoolIdx, consumersVotedForThisFormat in votes.items():
                if len(consumersVotedForThisFormat) == len(consumers):
                    tensor.format = producer.formatPool[producerFormatPoolIdx]
                    producer.formatPool = [tensor.format]

                    # Compare the tensor to the encloser, and make a mask of
                    # where stride is needed.
                    tshape = tensor.getShape()
                    eshape = tensor.getTopEncloserRecursive().getShape()
                    tensorAxisDemand = [
                        1 if x -
                        y != 0 else 0 for (
                            x,
                            y) in zip(
                            tshape,
                            eshape)]

                    # Readjust the formatPool of the consumers to reflect the
                    # choice taken
                    for consumerIdx, (_,
                                      consumerInputFmt) in consumersVotedForThisFormat:
                        consumers[consumerIdx].formatPool = matchConsumingFormat(
                            consumers[consumerIdx].formatPool,
                            consumerInputFmt,
                            tensor.getShape(),
                            tensorAxisDemand)
                    break

    """
        At this point, we have decided that G0 and G1 can be supported strided on any axis by producers, but cannot be strided by consumers.
    """

    """
        For every Op, that has more than one input tensor.... do something?
    """

    # Optimistically set the layout for tensors that are fed into layers with
    # multiple input tensors.
    for name in nx.lexicographical_topological_sort(g):
        layer_type = g.node[name]['type']

        if layer_type == 'OP':
            layer = g.node[name]['ref']

            # Skip layers that don't have multiple inputs
            if len(layer.getInputTensors()) <= 1:
                continue

            originalFormatPool = list(layer.formatPool)
            layerFormatUpdated = False

            for fmt in originalFormatPool:
                layer.formatPool = [fmt]

                counter = 0

                for tensor in layer.getInputTensors():
                    tensorName = tensor.getName().stringifyName()

                    producerNode = list(g.predecessors(tensorName))
                    assert(len(producerNode) == 1)
                    producer = g.node[producerNode[0]]['ref']

                    consumersNodes = list(g.successors(tensorName))
                    consumers = [g.node[name]['ref'] for name in consumersNodes]

                    votes = getVotes(producer, consumers, tensor)
                    for producerFormatPoolIdx, consumersVotedForThisFormat in votes.items():
                        if len(consumersVotedForThisFormat) == len(consumers):
                            counter += 1
                            break

                if counter == len(layer.getInputTensors()):
                    layerFormatUpdated = True

                    # Repeated code that updates the formats
                    for tensor in layer.getInputTensors():
                        tensorName = tensor.getName().stringifyName()

                        producerNode = list(g.predecessors(tensorName))
                        assert(len(producerNode) == 1)
                        producer = g.node[producerNode[0]]['ref']

                        consumersNodes = list(g.successors(tensorName))
                        consumers = [g.node[name]['ref']
                                     for name in consumersNodes]

                        # Optimistically set the tensor for layers that reach a
                        # consensus
                        votes = getVotes(producer, consumers, tensor)

                        for producerFormatPoolIdx, consumersVotedForThisFormat in votes.items():
                            if len(consumersVotedForThisFormat) == len(
                                    consumers):
                                tensor.format = producer.formatPool[producerFormatPoolIdx]
                                producer.formatPool = [tensor.format]

                                tshape = tensor.getShape()
                                eshape = tensor.getTopEncloserRecursive().getShape()
                                tensorAxisDemand = [
                                    1 if x -
                                    y != 0 else 0 for (
                                        x,
                                        y) in zip(
                                        tshape,
                                        eshape)]

                                # Readjust the formatPool of the consumers to
                                # reflect the choice taken
                                for consumerIdx, (_,
                                                  consumerInputFmt) in consumersVotedForThisFormat:
                                    consumers[consumerIdx].formatPool = matchConsumingFormat(
                                        consumers[consumerIdx].formatPool, consumerInputFmt, tensor.getShape(), tensorAxisDemand)
                                break
                    break
            if not layerFormatUpdated:
                layer.formatPool = originalFormatPool

    """
        For every tensor that has not been assigned a format, produce a Conversion
    """
    for name in nx.lexicographical_topological_sort(g):
        layer_type = g.node[name]['type']

        if layer_type == 'BLOB':
            tensor = g.node[name]['ref']

            # For the tensors that didn't reach a consensus, we use the minimum
            # number of conversion layers
            if not tensor.format:
                producerNode = list(g.predecessors(name))
                assert(len(producerNode) == 1)
                producer = g.node[producerNode[0]]['ref']

                consumersNodes = list(g.successors(name))
                consumers = [g.node[name]['ref'] for name in consumersNodes]

                # If there is a consumer that takes as input more than one input
                # tensor, skip it.
                skipTensor = False
                for consumer in consumers:
                    if len(consumer.getInputTensors()) > 1:
                        skipTensor = True
                        break
                if skipTensor:
                    print('Skipping', tensor.getName().stringifyName())
                    continue

                # Check if a format of the producer can satisfy some consumers.
                # If so, then peak the format that satisfies the most consumers.
                votes = getVotes(producer, consumers, tensor)

                # Set the tensor in this mode and insert conversion layers for
                # the rest.
                producerFormatPoolIdx, consumersIdxVotedForThisFormat = getPrevelantChoice(
                    votes)

                if producerFormatPoolIdx:

                    # print('Setting {} in mode {}'.format(layers, mode))
                    tensor.format = producer.formatPool[producerFormatPoolIdx]
                    producer.formatPool = [tensor.format]

                    tshape = tensor.getShape()
                    eshape = tensor.getTopEncloserRecursive().getShape()
                    tensorAxisDemand = [
                        1 if x -
                        y != 0 else 0 for (
                            x,
                            y) in zip(
                            tshape,
                            eshape)]

                    for consumerIdx, (_,
                                      consumerInputFmt) in votes[producerFormatPoolIdx]:
                        consumers[consumerIdx].formatPool = matchConsumingFormat(
                            consumers[consumerIdx].formatPool,
                            consumerInputFmt,
                            tensor.getShape(),
                            tensorAxisDemand)

                        # Mark as deleted
                        consumers[consumerIdx] = -1
                else:
                    # print(producer.getStringifiedName())
                    tensor.format = producer.formatPool[0]
                    producer.formatPool = [tensor.format]

                conversionLayerFmtPool = getDefaultConversionLayerFormatPool()

                votes = getVotesRaw(
                    conversionLayerFmtPool,
                    consumers,
                    tensor,
                    skipMarkedConsumers=True,
                    ignoreEnclosure=True)

                for mode, layers in minimizeConversionLayers(
                        OrderedDict(votes)):
                    # Add conversion layers (and tensors) between tensor and
                    # consumers.

                    # Create the Conversion layer and the tensor
                    convertedTensorName = '{}_converted'.format(
                        tensor.getName().stringifyOriginalName())
                    convertedTensor = UnpopulatedTensor(shape=tensor.getShape())
                    convertedTensor.setName(convertedTensorName)
                    # print('Producer', producer.getStringifiedName())

                    # TODO: Dont have None in conversionLayerFmtPool
                    convertedTensor.format = conversionLayerFmtPool[mode]

                    convertLayerName = "{}_convert".format(
                        producer.getName().stringifyOriginalName())
                    convertLayer = Conversion(
                        convertLayerName, [
                            tensor.getName()], [
                            convertedTensor.getName()])
                    convertLayer.setInputTensors([tensor])
                    convertLayer.setOutputTensors([convertedTensor])
                    convertLayer.loadInputTensorSizes([tensor.getShape()])
                    convertLayer.loadOutputTensorSizes(
                        [convertedTensor.getShape()])

                    convertLayer.deriveFormatPool(
                        g.node[name]['ref'].format[1], convertedTensor.format[1])

                    # Update the format of the consumers that read from the newly introduced conversion layer
                    # print('$$$', mode, votes)
                    tensorAxisDemand = [0, 0, 0, 0]

                    for consumerIdx, (_, consumerInputFmt) in votes[mode]:
                        consumers[consumerIdx].formatPool = matchConsumingFormat(
                            consumers[consumerIdx].formatPool,
                            consumerInputFmt,
                            tensor.getShape(),
                            tensorAxisDemand)

                    g.add_node(
                        convertLayer.getStringifiedName(),
                        type="OP",
                        ref=convertLayer)
                    g.add_node(
                        convertedTensor.getName().stringifyName(),
                        type="BLOB",
                        ref=convertedTensor)
                    g.add_edge(name, convertLayer.getStringifiedName())
                    g.add_edge(
                        convertLayer.getStringifiedName(),
                        convertedTensor.getName().stringifyName())

                    # Remove the edges from the tensor to the consumers
                    layerRefs = [consumers[i[0]] for i in layers]
                    for nodeName in list(g.successors(name)):
                        if g.node[nodeName]['ref'] in layerRefs:
                            g.remove_edge(name, nodeName)
                            g.add_edge(
                                convertedTensor.getName().stringifyName(), nodeName)

                            # TODO: Update the tensors that corresponds to these
                            # nodes
                            g.node[nodeName]['ref'].inputTensorNames = [
                                convertedTensor.getName()]
                            g.node[nodeName]['ref'].setInputTensors(
                                [convertedTensor])
                            g.node[nodeName]['ref'].loadInputTensorSizes(
                                [convertedTensor.getShape()])

    """
        For every tensor remaining that does not have a format, apply Conversions
    """

    # Add conversion layers for the rest (very inefficiently)
    for name in nx.lexicographical_topological_sort(g):
        layer_type = g.node[name]['type']

        if layer_type == 'BLOB':
            tensor = g.node[name]['ref']

            if not tensor.format:
                producerNode = list(g.predecessors(name))
                assert(len(producerNode) == 1)
                producer = g.node[producerNode[0]]['ref']

                consumersNodes = list(g.successors(name))
                consumers = [g.node[name]['ref'] for name in consumersNodes]

                # Select a format for the producer
                producer.formatPool = [producer.formatPool[0]]
                tensor.format = producer.formatPool[0]

                # Add a conversion for each consumer
                for consumer in consumers:
                    consumer.formatPool = [consumer.formatPool[0]]

                    # TODO: Improve this check
                    # Insert conversion layer only if necessary
                    if consumer.formatPool[0][0].layout == tensor.format[1].layout:
                        continue

                    convertedTensorName = '{}_converted'.format(
                        tensor.getName().stringifyOriginalName())
                    convertedTensor = UnpopulatedTensor(shape=tensor.getShape())
                    convertedTensor.setName(convertedTensorName)

                    # TODO: Dont have None in conversionLayerFmtPool
                    convertedTensor.format = (
                        producer.formatPool[0][1], consumer.formatPool[0][0])
                    # print('ConvertedTensorFormat', convertedTensor.format[0].getSignature(), convertedTensor.format[1].getSignature())

                    convertLayerName = "{}_convert".format(
                        producer.getName().stringifyOriginalName())
                    convertLayer = Conversion(
                        convertLayerName, [
                            tensor.getName()], [
                            convertedTensor.getName()])

                    convertLayer.deriveFormatPool(
                        g.node[name]['ref'].format[1], convertedTensor.format[1])

                    successors_before_mutation = list(g.successors(name))

                    # Add a conversion layer (and the converted tensor) between the tensor `name`
                    # and the operation `consumer`.
                    g.add_node(
                        convertLayer.getStringifiedName(),
                        type="OP",
                        ref=convertLayer)
                    g.add_node(
                        convertedTensor.getName().stringifyName(),
                        type="BLOB",
                        ref=convertedTensor)
                    g.remove_edge(name, consumer.getStringifiedName())
                    g.add_edge(name, convertLayer.getStringifiedName())
                    g.add_edge(
                        convertLayer.getStringifiedName(),
                        convertedTensor.getName().stringifyName())
                    g.add_edge(
                        convertedTensor.getName().stringifyName(),
                        consumer.getStringifiedName())

                    # Store the edge information in the layers
                    convertLayer.setInputTensors([tensor])
                    convertLayer.setOutputTensors([convertedTensor])
                    convertLayer.loadInputTensorSizes([tensor.getShape()])
                    convertLayer.loadOutputTensorSizes(
                        [convertedTensor.getShape()])
                    # Search and replace the tensor in the consumer
                    # Find the index where this nodes has tensor whose name is
                    # `name`
                    idx = -1
                    for i, tensor in enumerate(consumer.getInputTensors()):
                        if tensor.getName().stringifyName() == name:
                            idx = i
                            break
                    assert(idx >= 0)

                    inputTensors = list(consumer.getInputTensors())
                    inputTensors[idx] = convertedTensor

                    consumer.inputTensorNames = [
                        t.getName() for t in inputTensors]
                    consumer.setInputTensors(inputTensors)

                    inputTensorShapes = list(consumer.getInputTensorSizes())
                    inputTensorShapes[idx] = convertedTensor.getShape()
                    consumer.loadInputTensorSizes(inputTensorShapes)

                    print('Adding edge between', name, 'and',
                          convertLayer.getStringifiedName())
                    print(
                        'Adding edge between',
                        convertLayer.getStringifiedName(),
                        'and',
                        convertedTensor.getName().stringifyName())
                    print(
                        'Successors before mutation',
                        successors_before_mutation)

    # Set the layout for the tensors
    for name in nx.lexicographical_topological_sort(g):
        if g.node[name]['type'] == 'BLOB':
            tensor = g.node[name]['ref']

            cons, prod = tensor.format
            if prod:
                # print('Setting producer ({}, {}) to {}'.format(tensor.getName().stringifyName(), tensor.ID, prod.layout))
                tensor.setLayout(prod.layout)
            elif cons:
                # print('Setting consumer ({}, {}) to {}'.format(tensor.getName().stringifyName(), tensor.ID, cons.layout))
                tensor.setLayout(cons.layout)

    # TODO: Improve to the general case (Multiple producers and consumers)
    # For each tensor, we look at the producer and the consumer and place the tensor
    # inside something else that satisfies both.

    # Set the layout for the tensors
    for name in nx.lexicographical_topological_sort(g):
        if g.node[name]['type'] == 'BLOB':
            tensor = g.node[name]['ref']

            producersNodes = list(g.predecessors(name))
            producers = [g.node[name]['ref'] for name in producersNodes]

            consumersNodes = list(g.successors(name))
            consumers = [g.node[name]['ref'] for name in consumersNodes]

            # print('Producer: {}, Consumer: {}'.format(
            #     producersNodes[0], consumersNodes[0]))

            newShape = producers[0].formatPool[0][1].ensureCompatibility(
                consumers[0].formatPool[0][0], tensor.getShape())

            if newShape != tensor.getShape():
                newTensor = UnpopulatedTensor(shape=newShape)
                print(
                    '\tPlace {} ({}) in {}'.format(
                        tensor.getName().stringifyName(),
                        tensor.getShape(),
                        newTensor.getShape()))

                newTensor.setName(tensor.name.stringifyOriginalName())
                newTensor.setLayout(tensor.getLayout())
                newTensor.setDatatype(tensor.dtype)
                tensor.place(newTensor, Tensor.ORIGIN)

    # TODO: (Remove the following hack)
    # Priorbox is a special layer where only the shape of the input is required.
    # For now, the is a limitation, because the type TensorFormat does not support
    # the following things: i) Layers with multiple inputs with different layout per input.
    # ii) More importantly, layers that accept any layout (i.e. any is a
    # special format type).
    from Controllers.Parsers.Parser.PriorBox import PriorBox

    # Detect if the input of a priorbox comes from conversion layer. If so, and
    # the output of the conversion layer is only consumed by priorboxes, then set
    # set conversion layer to implicit.
    for name in nx.lexicographical_topological_sort(g):
        if g.node[name]['type'] == 'OP':
            op = g.node[name]['ref']

            if isinstance(op, Conversion):
                # Get the consumers of the output tensor
                consumersNodes = list(g.successors(name))
                assert(len(consumersNodes) == 1)

                tensorName = consumersNodes[0]
                tensor = g.node[tensorName]['ref']

                consumersNodes = list(g.successors(tensorName))
                consumers = [g.node[name]['ref'] for name in consumersNodes]

                # If all consumers of the converted tensor are PriorBoxes,
                # set the conversion layer to be implicit
                priorboxCount = 0
                for consumer in consumers:
                    if isinstance(consumer, PriorBox):
                        priorboxCount += 1

                if priorboxCount == len(consumers):
                    op.setImplicit()
                elif len(consumers) == 1:  # Remove useless conversion layer
                    producersTensors = list(g.predecessors(name))
                    if len(producersTensors) == 1 and all([findSolution(
                            fp, consumers[0], g.node[producersTensors[0]]['ref']) for fp in consumers[0].formatPool]):
                        print(
                            "Remove unecessary conversion layer {}".format(name))
                        op.setImplicit()
                        op.getInputTensors()[0].setLayoutRecursivly(
                            op.getOutputTensors()[0].getLayout())
                        op.getInputTensors()[0].getTopEncloserRecursive().place(
                            op.getOutputTensors()[0].getTopEncloserRecursive(), (0, 0, 0, 0))

    return buildLayerLists(g)
