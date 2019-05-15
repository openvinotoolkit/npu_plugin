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
from ctypes import *
from Controllers.MiscIO import *
from Controllers.FileIO import *
from Models.EnumDeclarations import *
from Controllers.BlobBuilder import *
import sys
import collections

from Controllers.PingPong import getManualHwSchedule, detectPermutationOrder
import Controllers.Globals as GLOBALS

wlink = WeakLinkManager()


class Blob:
    def __init__(
            self,
            version,
            name,
            report_dir,
            myriad_params,
            network,
            blob_name):
        """
        This object contains all the information required for a blob file + some additional info for processing.
        :param version: The version of the toolkit used to generate this blob. Useful for the potential of
         having backwards compatibility - although it's easier to regenerate your file.
        :param name: Name of the network represented in the blob file
        :param report_dir: Where to output our reports (Note: TODO)
        :param myriad_params: Myriad configurations (Note: TODO)
        :param network: A Network object to attach to the blob.
        :return:
        """
        self.major_version = c_uint32(version[0])
        self.minor_version = c_uint32(version[1])
        self.patch_version = c_uint32(version[2])
        self.filesize = c_uint32(16)
        self.name = set_string_range(name, 100).encode('ascii')
        self.report_dir = set_string_range(report_dir, 100).encode('ascii')
        self.myriad_params = myriad_params
        self.network = network
        self.stage_count = c_uint32(self.network.count)
        self.VCS_Fix = True
        self.blob_name = blob_name

    def generate_v2(self, args):
        """
        Generates a blob file complying to version 2 design document.
        :return:  None. Creates file to src folder.
        """

        # General Assignments
        stage_list = self.network.stageslist
        self.network.bundle()    # Creates Buffer Objects.

        global wlink

        # Global objects is import * pollutes the namespace in very
        # strange ways. These need to be removed immediately!
        from Controllers.FileIO import buffer

        # Creation
        blob = Container()

        header = Container(align=16)

        elf_header = Container()
        write_elf_header(elf_header)
        header.meld(elf_header)

        header.push("magic_number", Value(c_uint32(8708)))
        header.push("file_size",    Placeholder(c_uint32(16)))

        header.push("blob_major_version", Value(self.major_version))
        header.push("blob_minor_version", Value(self.minor_version))
        # TODO: Patch Version.
        header.push("num_shaves", Value(c_uint32(args.number_of_shaves)))

        header.push("stage_section_offset", Placeholder(c_uint32(16)))
        header.push("buffer_section_offset", Placeholder(c_uint32(16)))
        header.push("relocation_offset", Placeholder(c_uint32(16)))


        header.push("size_of_input",  Value(c_uint32(np.prod(self.network.inputTensor.shape))))

        pingPongPair = getManualHwSchedule()

        if pingPongPair is not None and pingPongPair.enable_perm:
            permutedBasicBlocks = detectPermutationOrder(stage_list)

            header.push("permutation_enabled",  Value(c_uint32(1)))
            perm_sz = sum([len(sublist) for sublist in permutedBasicBlocks])
            header.push("permutation_size",  Value(c_uint32(perm_sz)))
            for b in permutedBasicBlocks:
                for name, id in b:
                    header.push("permutation_{}".format(name),  Value(c_uint32(id)))
        else:
            header.push("permutation_enabled",  Value(c_uint32(0)))

        header.update("stage_section_offset", Value(c_uint32(header.aligned_size)))


        stageSectionHeader = Container(align=16)
        stageSectionHeader.push("stage_count",      Value(c_uint32(len(stage_list))))
        stageSectionHeader.push("size_of_stage_section",  Placeholder(c_uint32(16)))
        stageSectionHeader.push("size_of_output",  Value(c_uint32(np.prod(self.network.outputTensorShape))))

        # Create an array of Stages
        for idx, x in enumerate(stage_list):
            s = Container()

            s.push("next_stage", Placeholder(c_uint32(16)))

            s.push("stage_type", Value(c_uint32(x.op.value)))

            s.push("implementation", Value(c_uint32(x.optMask)))

            specific_stage_parse(s, x)
            s.push("preop_type", Value(c_uint32(x.preOp.value)))
            specific_stage_parse(s, x, subops=1) # Do postop and preops
            s.push("postop_type", Value(c_uint32(x.postOp.value)))
            specific_stage_parse(s, x, subops=2) # Do postop and preops


            if idx != len(stage_list) -1:
                s.update("next_stage", StableLink(c_uint32(stageSectionHeader.size + s.aligned_size)))
            else:
                # end, no next node.
                s.update("next_stage", StableLink(c_uint32(0)))

            stageSectionHeader.meld(s)



        stageSectionHeader.update("size_of_stage_section", Value(c_uint32(stageSectionHeader.aligned_size)))

        header.update("buffer_section_offset", Value(c_uint32(header.aligned_size + stageSectionHeader.aligned_size)))

        bufferSectionHeader = Container()
        bufferSectionHeader.push("size_of_buffer_section", Placeholder(c_uint32(16)))
        for x in range(3):  # TODO: REMOVE
            bufferSectionHeader.push("Pad"+str(x), Value(c_uint32(42)))

        bufferSection = Container()
        global buffer
        for idx, x in enumerate(buffer):
            bufferSection.push("buffer"+str(idx), BinaryData(x))

        bufferSectionHeader.update("size_of_buffer_section", Value(c_uint32(bufferSectionHeader.aligned_size + bufferSection.aligned_size)))
        bufferSectionHeader.meld(bufferSection)

        header.update("relocation_offset", Value(c_uint32(header.aligned_size + stageSectionHeader.aligned_size + bufferSectionHeader.aligned_size)))
        relocationSectionHeader = Container()
        relocationSectionHeader.push("relocation_buffer_size",   Placeholder(c_uint32(99999)))
        relocationSectionHeader.push("blob_buffer_reloc_offset", Placeholder(c_uint32(99999)))
        relocationSectionHeader.push("blob_buffer_reloc_size",   Placeholder(c_uint32(99999)))
        relocationSectionHeader.push("work_buffer_reloc_offset", Placeholder(c_uint32(99999)))
        relocationSectionHeader.push("work_buffer_reloc_size",   Placeholder(c_uint32(99999)))

        blobRelocationSection = Container()
        # Blob Buffer
        for idx,x in enumerate((y for y in wlink.link_array if y.locale.value == MemoryIndex.blob.value)):
            blobRelocationSection.push(str(idx) + "_blob_link_off", Value(x.val))
            blobRelocationSection.push(str(idx) + "_blob_link_loc", Value(x.locale))

        relocationSectionHeader.update("blob_buffer_reloc_offset", Value(c_uint32(header.aligned_size + stageSectionHeader.aligned_size + bufferSectionHeader.aligned_size + relocationSectionHeader.aligned_size)))
        relocationSectionHeader.update("blob_buffer_reloc_size",   Value(c_uint32(blobRelocationSection.aligned_size)))
        relocationSectionHeader.meld(blobRelocationSection)


        bssRelocationSection = Container()
        # BSS Buffer

        for idx,x in enumerate((y for y in wlink.link_array if  y.locale.value >= MemoryIndex.workbuffer.value)):
            bssRelocationSection.push(str(idx) + "_work_link_off", Value(x.val))
            bssRelocationSection.push(str(idx) + "_work_link_loc", Value(x.locale))

        relocationSectionHeader.update("work_buffer_reloc_offset", Value(c_uint32(header.aligned_size + stageSectionHeader.aligned_size  + bufferSectionHeader.aligned_size + relocationSectionHeader.aligned_size)))
        relocationSectionHeader.update("work_buffer_reloc_size",   Value(c_uint32(bssRelocationSection.aligned_size)))
        relocationSectionHeader.meld(bssRelocationSection)

        relocationSectionHeader.update("relocation_buffer_size", Value(c_uint32(relocationSectionHeader.aligned_size)))

        header.update("file_size",  Value(c_uint32(header.aligned_size + stageSectionHeader.aligned_size  + bufferSectionHeader.aligned_size + relocationSectionHeader.aligned_size)))
        blob.meld(header)
        blob.meld(stageSectionHeader)
        blob.meld(bufferSectionHeader)
        blob.meld(relocationSectionHeader)

        # blob.print(include_filters=["stage_type"])
        # blob.print()  # Should not be enabled in Release/Main Branch. Debug Only.


        with open(self.blob_name, 'wb') as f:
            blob.write(f)

        print("Blob generated")

class BlobFile():
    # TODO: Is this used?
    def __init__(self):
        self.contents = []

    def push(self, item):
        """
        Used to push sections onto the doc.
        :param item:
        :return:
        """
        self.contents.append(item)

    def write(self, f):
        for field in self.contents:
            field.write(f)

    def print(self):
        for x in self.contents:
            x.print()

class BlobFieldContainer():
    # TODO: Is this used?
    def __init__(self):
        self.attr = collections.OrderedDict()

    def push(self, label, item):
        self.attr[label] = item
        # print("["+label+"] =", item)


    def change(self, label, item):
        assert label in self.attr, "Cannot change field that does not exist yet."
        self.attr[label] = item
        # print("["+label+"] <=", item)

    def size(self):
        sz = 0

        for t in self.attr:
            val = self.attr[t]
            if isinstance(val, ctypes._SimpleCData) or isinstance(val, bytes):
                sz += byte_size(val)
            elif isinstance(val, BlobFieldContainer):
                sz += val.size()

        return sz

    def write(self, f):

        for t in self.attr:
            val = self.attr[t]
            if isinstance(val, ctypes._SimpleCData) or isinstance(val, bytes):
                f.write(val)
            elif isinstance(val, BlobFieldContainer):
                val.write(f)

    def print(self):
        for k in self.attr:
            print(k)


def specific_stage_parse(stageCon, netObj, subops=0):

    if subops == 1: # PreOps
        if netObj.preOp == StageType.convolution:  # Code 0
            netObj.preOp = StageType.none
            return
        elif netObj.preOp in [StageType.storage_order_convert]:
            netObj.pre_definition.specific_details_push(stageCon, netObj)

    if subops == 0: # Ops
        netObj.definition.specific_details_push(stageCon, netObj)
    if subops == 2: # Post Ops
        if netObj.postOp == StageType.convolution:
            netObj.postOp = StageType.none
        elif netObj.postOp in [StageType.relu,
            StageType.leaky_relu,
            StageType.relu_x,
            StageType.storage_order_convert]:
            netObj.post_definition.specific_details_push(stageCon, netObj)

def helper_parseBuffer(prefix, obj, data):
    global wlink

    obj.push(prefix+"DimX", Value(c_uint32(data.x if data.x is not None else 0)))
    obj.push(prefix+"DimY", Value(c_uint32(data.y if data.y is not None else 0)))
    obj.push(prefix+"DimZ", Value(c_uint32(data.z if data.z is not None else 0)))

    obj.push(prefix+"StrideX", Value(c_uint32(data.x_s if data.x_s is not None else 0)))
    obj.push(prefix+"StrideY", Value(c_uint32(data.y_s if data.y_s is not None else 0)))
    obj.push(prefix+"StrideZ", Value(c_uint32(data.z_s if data.z_s is not None else 0)))

    # obj.push(prefix+"BufferLocation", Value(c_uint32()))

    offset = data.offset if data.offset is not None else 0
    location = data.location if data.location is not None else 0

    wl = wlink.new(c_uint32(offset), c_uint32(location))

    obj.push(prefix+"Offset", Value(c_uint32(wlink.index(wl))))
    obj.push(prefix+"Location", Value(c_uint32(location)))
    obj.push(prefix+"dataType", Value(c_uint32(data.dtype if data.dtype is not None else 0)))
    obj.push(prefix+"order", Value(c_uint32(data.order.value if data.order.value is not None else 0)))


def write_elf_header(area):
    e_ident = "0x7F454C46"                    # ELFMAG
    e_ident += "01"                           # EI_CLASS
    e_ident += "01"                           # EI_DATA (Not-2's compliment)
    e_ident += "01"                           # EI_CURRENT
    e_ident += "000000000000000000"           # EI_PAD
    area.push("e_ident",     Value(c_uint16(int(e_ident, 16))))
    area.push("e_type",      Value(c_uint16(1)))
    area.push("e_machine",   Value(c_uint16(2)))
    area.push("e_version",   Value(c_uint16(1)))
    area.push("e_entry",     Value(c_uint32(0)))
    area.push("e_phoff",     Value(c_uint32(0)))
    area.push("e_shoff",     Value(c_uint32(0)))
    area.push("e_flags",     Value(c_uint16(0)))
    area.push("e_ehsize",    Value(c_uint16(272)))
    area.push("e_phentsize", Value(c_uint16(0)))
    area.push("e_phnum",     Value(c_uint16(0)))
    area.push("e_shentsize", Value(c_uint16(0)))
    area.push("e_shnum",     Value(c_uint16(0)))
    area.push("e_shstrndx",  Value(c_uint16(0)))

#    print("ELF Size: ", area.size)
