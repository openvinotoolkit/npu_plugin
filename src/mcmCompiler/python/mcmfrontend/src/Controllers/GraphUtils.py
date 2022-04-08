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


import os
import tempfile
import networkx as nx
import pygraphviz as pgv

# TODO: Move to Views


def buildGraph(parsedLayers, contract_blobs=True):

    if not nx.__version__.startswith('2'):
        raise ValueError(
            "Unsupported NetworkX library. Please upgrade it (pip3 install networkx --upgrade) ")

    # Create empty graph. We want to preserve the order of parents/children
    g = nx.OrderedDiGraph()

    # Add transformations as nodes and connect them with blobs
    for layer in parsedLayers:
        # add blob connections
        for idx, inputTensorName in enumerate(layer.getInputTensorNames()):
            ref = layer.inputTensors[idx] if hasattr(
                layer, 'inputTensors') else None
            g.add_node(inputTensorName.stringifyName(), type="BLOB", ref=ref)
        for idx, outputTensorName in enumerate(layer.getOutputTensorNames()):
            ref = layer.outputTensors[idx] if hasattr(
                layer, 'outputTensors') else None
            g.add_node(outputTensorName.stringifyName(), type="BLOB", ref=ref)

    for layer in parsedLayers:
        g.add_node(layer.getStringifiedName(), type="OP", ref=layer)
        # Add in/out connections
        for inputTensorName in layer.getInputTensorNames():
            g.add_edge(
                inputTensorName.stringifyName(),
                layer.getStringifiedName())
        for outputTensorName in layer.getOutputTensorNames():
            g.add_edge(
                layer.getStringifiedName(),
                outputTensorName.stringifyName())

    # Contract the blob into the operation
    if contract_blobs:
        g = contractBlobs(g)

    return g


def buildLayerLists(g, contract_blobs=True):
    scheduler_layer = []
    for stageName in nx.lexicographical_topological_sort(g):
        if g.node[stageName]['type'] == 'OP':
            scheduler_layer.append(g.node[stageName]['ref'])

    return scheduler_layer


def contractBlobs(g):
    for n in g.node:
        if g.node[n]['type'] == 'BLOB':
            for parent in g.predecessors(n):
                g = nx.contracted_edge(g, (parent, n), self_loops=False)
                g.node[parent].pop('contraction', None)
    return g
