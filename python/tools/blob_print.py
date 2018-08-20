import construct
from construct import *
from construct import Struct
import ctypes
import argparse


from ctypes import c_uint32
from enum import Enum
import numpy as np

from collections import OrderedDict

HEX = '0x{:08X}'


class HwDescOp(Enum):
    """
    These are the enums used in the hardware to identify layers.
    """
    convolution = 0
    convolution_with_pooling = 1
    fully_connected_convolution = 2
    pooling_only = 4


class SerializedDescriptor:
    def __init__(self, type_):
        self.layout = OrderedDict()

        descriptor_header = OrderedDict([
            # Line 0
            ("NextDesc", [32, None, HEX]),
            ("Type", [3, None]),
            ("mode", [3, None]),
            ("rsvd_01_interleavedInput", [1, None]),
            ("rsvd_01_interleavedOutput", [1, None]),
            ("id", [4, None]),
            ("it", [4, None]),
            ("cm", [3, None]),
            ("dm", [1, None]),
            ("disaint", [1, None]),
            ("rsvd_02", [11, None]),
        ])
        self.layout.update(descriptor_header)

        if type_ == "Conv":
            main_layout = OrderedDict([
                # Line 1
                ("iDimY-1", [12, None]),
                ("rsvd_10_topOutputJunk", [4, None]),
                ("iDimX-1", [12, None]),
                ("rsvd_11_bottomOutputJunk", [4, None]),
                ("iChans-1", [11, None]),
                ("rsvd_12", [5, None]),
                ("oChans-1", [11, None]),
                ("rsvd_13", [3, None]),
                ("rsvd_13_interleaved", [2, None]),

                # Line 2
                ("ChRamBlk-1", [11, None]),
                ("rsvd_20", [5, None]),
                ("Ch_stride", [4, None]),
                ("rsvd_21", [12, None]),
                ("InFw-1", [4, None]),
                ("InFh-1", [4, None]),
                ("rsvd_22", [19, None]),
                ("PadType", [4, None]),
                ("PadEnable", [1, None]),

                # Line 3
                ("poolEn", [1, None]),
                ("rsvd7", [15, None]),
                ("poolKernelHeight-1", [8, None]),
                ("poolKernelWidth-1", [8, None]),
                ("avgPoolX", [16, None]),
                ("rsvd8", [15, None]),
                ("poolType", [1, None]),

                # Line 4
                ("dataBaseAddr", [32, None, HEX]),
                ("t0", [10, None]),
                ("a0", [10, None]),
                ("a1", [10, None]),
                ("reluXEn", [1, None]),
                ("reluEn", [1, None]),

                # Line 5
                ("dataChStr", [32, None]),
                ("dataLnStr", [32, None]),

                # Line 6
                ("coeffBaseAddr", [32, None, HEX]),
                ("coeffChStrOut", [32, None]),

                # Line 7
                ("coeffChStrIn", [32, None]),
                ("outLnStr", [32, None]),

                # Line 8
                ("outBaseAddr", [32, None, HEX]),
                ("outChStr", [32, None]),

                # Line 9
                ("localLs", [9, None]),
                ("rsvd_90", [7, None]),
                ("localCs", [13, None]),
                ("rsvd_91", [3, None]),
                ("linesPerCh-1", [9, None]),
                ("rsvd_92_SODGroup", [22, None]),
                ("rud", [1, None]),

                # Line 10
                ("minLines-1", [9, None]),
                ("rsvd_A0__SOHGroup", [23, None]),
                ("coeffLpb-1", [8, None]),
                ("css-1", [8, None]),
                ("outputX", [12, None]),
                ("rsvd_A1", [4, None]),

                # Line 11
                ("biasBaseAddr", [32, None, HEX]),
                ("scaleBaseAddr", [32, None, HEX]),
            ])

        elif type_ == "Pool":
            main_layout = OrderedDict([

                # Line 1
                ("iDimY-1", [12, None]),
                ("rsvd_10", [4, None]),
                ("iDimX-1", [12, None]),
                ("rsvd_11", [4, None]),
                ("iChans-1", [11, None]),
                ("rsvd_12", [5, None]),
                ("oChans-1", [11, None]),
                ("rsvd_13", [3, None]),
                ("interleaved", [2, None]),

                # Line 2
                ("ChRamBlk-1", [11, None]),
                ("rsvd_20", [5, None]),
                ("stride", [4, None]),
                ("rsvd_21", [12, None]),
                ("InFw-1", [4, None]),
                ("InFh-1", [4, None]),
                ("rsvd_22", [19, None]),
                ("PadType", [4, None]),
                ("PadEnable", [1, None]),

                # Line 3
                ("poolEn", [1, None]),
                ("rsvd7", [15, None]),
                ("poolKernelHeight-1", [8, None]),
                ("poolKernelWidth-1", [8, None]),
                ("avgPoolX", [16, None]),
                ("rsvd8", [15, None]),
                ("poolType", [1, None]),

                # Line 4
                ("dataBaseAddr", [32, None, HEX]),
                ("t0", [10, None]),
                ("a0", [10, None]),
                ("a1", [10, None]),
                ("reluXEn", [1, None]),
                ("reluEn", [1, None]),

                # Line 5
                ("dataChStr", [32, None]),
                ("dataLnStr", [32, None]),

                # Line 6
                ("rsvd_60", [32, None]),
                ("rsvd_61", [32, None]),

                # Line 7
                ("rsvd_70", [32, None]),
                ("outLnStr", [32, None]),

                # Line 8
                ("outBaseAddr", [32, None, HEX]),
                ("outChStr", [32, None]),

                # Line 9
                ("localLs", [9, None]),
                ("rsvd_90", [7, None]),
                ("localCs", [13, None]),
                ("rsvd_91", [3, None]),
                ("linesPerCh-1", [9, None]),
                ("rsvd_92", [22, None]),
                ("rud", [1, None]),

                # Line 10
                ("minLines-1", [9, None]),
                ("rsvd_A0", [23, None]),
                ("rsvd_A1", [8, None]),
                ("rsvd_A2", [8, None]),
                ("outputX", [12, None]),
                ("rsvd_A3", [4, None]),

                # Line 11
                ("biasBaseAddr", [32, None, HEX]),
                ("scaleBaseAddr", [32, None, HEX]),
            ])

        elif type_ == "FCL":
            main_layout = OrderedDict([

                # Line 1
                ("iDimX-1",  [12, None]),      # InputWidth
                ("rsvd0",    [20, None]),
                ("iChans-1", [8, None]),     # Vectors
                ("rsvd1",    [8, None]),
                ("oChans-1", [8, None]),      # vectors2
                ("rsvd2",    [8, None]),

                # Line 2
                ("ChRamBlk-1",   [9, None]),   # dataPerRamBlock
                ("rsvd3",        [23, None]),
                ("rsvd4",        [32, None]),

                # Line 3
                ("rsvd5",                [1, None]),
                ("actualOutChannels",    [8, None]),  # Custom Info
                ("rsvd5_",                [23, None]),
                ("X",                    [16, None]),
                ("rsvd6",                [16, None]),

                #Line 4
                ("dataBaseAddr",     [32, None, HEX]),
                ("t0",               [10, None]),
                ("a0",               [10, None]),
                ("a1",               [10, None]),
                ("reluXEn",          [1, None]),
                ("reluEn",           [1, None]),

                # Line 5
                ("dataChStr", [32, None]),
                ("dataLnStr", [32, None]),

                # Line 6
                ("vecBaseAddr", [32, None, HEX]),
                ("vecStrOut", [32, None, HEX]),

                # Line 7
                ("vecStrIn", [32, None]),
                ("outLnStr", [32, None]),

                # Line 8
                ("outBaseAddr", [32, None, HEX]),
                ("outChStr", [32, None]),

                # Line 9
                ("localLs", [9, None]),
                ("rsvd7", [7, None]),
                ("localBs", [13, None]),
                ("rsvd8", [3, None]),
                ("rsvd9", [31, None]),
                ("rud", [1, None]),

                # Line 10
                ("rsvd10", [16, None]),
                ("Acc", [1, None]),
                ("rsvd11", [15, None]),
                ("vecLPB-1", [8, None]),
                ("rsvd12", [8, None]),
                ("outputX", [12, None]),
                ("rsvd12_", [4, None]),

                # Line 11
                ("biasBaseAddr", [32, None, HEX]),
                ("scaleBaseAddr", [32, None, HEX]),
            ])
        else:
            assert 0, "Invalid Descriptor."

        self.layout.update(main_layout)

        pallete_fields = OrderedDict([
            # Line 12
            ("p0", [16, None]),
            ("p1", [16, None]),
            ("p2", [16, None]),
            ("p3", [16, None]),

            ("p4", [16, None]),
            ("p5", [16, None]),
            ("p6", [16, None]),
            ("p7", [16, None]),

            ("p8", [16, None]),
            ("p9", [16, None]),
            ("pA", [16, None]),
            ("pB", [16, None]),

            ("pC", [16, None]),
            ("pD", [16, None]),
            ("pE", [16, None]),
            ("pF", [16, None]),
        ])

        self.layout.update(pallete_fields)

    def set_field(self, field, value):
        # print(field, value)
        assert field in self.layout, "No such field in Descriptor"
        assertBitSize(value, self.layout[field][0])
        self.layout[field][1] = value

    def set_pallete(self, arr_16_elements):
        """
        helper function to set the palletized weights. If None, fill with zeroes.
        :param arr_16_elements: ordered 16 element array to populate pallete.
        :return: N/A
        """

        if arr_16_elements == None:
            arr_16_elements = [0]*16
        else:
            assert len(arr_16_elements) == 16, "Pallete not fully set."
        self.set_field("p0", arr_16_elements[0])
        self.set_field("p1", arr_16_elements[1])
        self.set_field("p2", arr_16_elements[2])
        self.set_field("p3", arr_16_elements[3])
        self.set_field("p4", arr_16_elements[4])
        self.set_field("p5", arr_16_elements[5])
        self.set_field("p6", arr_16_elements[6])
        self.set_field("p7", arr_16_elements[7])
        self.set_field("p8", arr_16_elements[8])
        self.set_field("p9", arr_16_elements[9])
        self.set_field("pA", arr_16_elements[10])
        self.set_field("pB", arr_16_elements[11])
        self.set_field("pC", arr_16_elements[12])
        self.set_field("pD", arr_16_elements[13])
        self.set_field("pE", arr_16_elements[14])
        self.set_field("pF", arr_16_elements[15])

    def print(self):
        prev_line = -1
        this_line = 0
        bit_count = 0
        for x in self.layout:
            if prev_line != this_line:
                # New Line
                print("XLine " + str(this_line), end=": ")
                prev_line = this_line
                bit_count = 0
            if len(self.layout[x]) == 3:
                print(x, self.layout[x][2].format(self.layout[x][1]), end='. ')
            else:
                print(x, self.layout[x][1], end='. ')
            bit_count += self.layout[x][0]
            if bit_count >= 64:
                # End of Line
                this_line+=1
                print("")

        print("")   # Newline

    def deserialize(self, desc):
        """
        Takes in a hardware descriptor and creates an instance of this class with it.
        Currently only supports Conv.

        :param desc: Descriptor, can include 0x and whitespace.
        :return: N/A
        """
        # Clean Descriptor
        desc = desc.replace("0x","")
        desc = ''.join(desc.split()) # Strip whitespace characters

        lines = []

        x = 0
        step = 8
        hl = 0
        while x < len(desc):                            # Each 'half-line' is 32 bits, or 8 hex values.
            hex_ = desc[x:x+step]                       # We iterate over each of these half-lines.
            if hl % 2:
                nl = "\n"
            else:
                nl = ""
            print(hex_, end=nl)
            bin_ = bin(int(hex_,16))[2:]
            # print("orig:   ", bin_)
            # bin_ = bin_[::-1].zfill(step*4)[::-1]
            bin_ = bin_.zfill(step*4)
            # print("filled: ",bin_)

            lines.append(bin_)                          # and add to a larger list.
            x+=step
            hl += 1

        char_count = 0

        for field in self.layout:
            field_bits = self.layout[field][0]          # How many bits required for field.


            idx = (char_count)//32                      # Index of Half-line in line list.
            target_line = lines[idx]

            # target_line = target_line[::-1]

            # if field == "dataBaseAddr":
            #     print("Original Binary: ", target_line, "To Extract", field, "from", field_bits, "bits. On half-line:", idx)

            tmp_cc = char_count - ((char_count//32)*32)


            # We then index into the half-line from the other direction.
            end = 32-tmp_cc
            start = 32-(tmp_cc + field_bits)

            # end = (tmp_cc + field_bits)
            # start = tmp_cc

            target_segment = target_line[start:end]
            # if field in ["dataBaseAddr"]:
            #     print("Extracted: ", target_segment, "@", start, ":", end)
            cut_value = target_segment.zfill(field_bits)      # Strip the part we want and pad with 0s if required.

            char_count += field_bits                    # Increment pointer

            # if field == "ChRamBlk-1":
            #     print(field, cut_value, '0x{:08X}'.format(int(cut_value,2)), int(cut_value,2))
            self.set_field(field, int(cut_value,2))     # Populate field


    def serialize(self):
        lines = self.serialize_lines()
        bk = []
        for x in lines:
            bk.append(c_uint32(x))

        return bk


    def serialize_lines(self, debug=False):
        prev_line = -1
        this_line = 0
        bit_count = 0

        cnt = []
        dsc = 0

        for index, x in enumerate(self.layout):
            if prev_line != this_line:
                # New Line
                prev_line = this_line
                bit_count = 0
                dsc = 0

            if self.layout[x][1] is None and "rsvd" in x:
                self.layout[x][1] = 0
                if debug:
                    print("Warning: Reserved Field Defaulted to Zero")
            elif self.layout[x][1] is None:
                print("Error: Required Field not populated:", x)
                quit()

            dsc += (self.layout[x][1] << bit_count)
            bit_count += self.layout[x][0]

            if bit_count >= 32 or (index == len(self.layout) - 1):
                # End of Line
                this_line+=1
                cnt.extend([dsc])

        return cnt


def assertBitSize(value, allowed_bits):
    """
    Ensures that when written to a field, does not overflow boundaries
    :param value: The value we want to enter into the space
    :param allowed_bits: size of the space for this field, in bits.
    :return: N/A
    """

    # print("Values allowed 0-"+str(2**(allowed_bits)), ". Value: ", value)
    if(type(value) not in [int, bool, np.int64, np.uint16, np.uint32, np.int32, np.uint64, np.int16]):
        print("field is not int")
        assert(0)
    if 2**(allowed_bits) <= value:
        print("field overflow")
        assert(0)
    if 0 > value:
        print("field underflow")
        assert(0)


# Expose for any module consumers
blob_format = Struct(
    # Elf Header
    "e_ident" / Int16ul,
    "e_type" / Int16ul,
    "e_machine" / Int16ul,
    "e_version" / Int16ul,
    "e_entry" / Int32ul,
    "e_phoff" / Int32ul,
    "e_shoff" / Int32ul,
    "e_flags" / Int16ul,
    "e_ehsize" / Int16ul,
    "e_phentsize" / Int16ul,
    "e_phnum" / Int16ul,
    "e_shentsize" / Int16ul,
    "e_shnum" / Int16ul,
    "e_shstrndx" / Int16ul,
    "magic_blob" / Int32ul,

    "file_size" / Int32ul,
    "blob_major_version" / Int32ul,
    "blob_minor_version" / Int32ul,
    "num_shaves" / Int32ul,
    "stage_section_offset" / Int32ul,
    "buffer_section_offset" / Int32ul,
    "relocation_offset" / Int32ul,
    "size_of_input" / Int32ul,
    # TODO: Hardware
    "Permutation_Enabled" / Int32ul,
    "ipad0" / Byte,
    "ipad1" / Byte,
    "ipad2" / Byte,
    "ipad3" / Byte,
    "ipad4" / Byte,
    "ipad5" / Byte,

    "stage_count" / Int32ul,
    "size_of_stage_section" / Int32ul,
    "size_of_output" / Int32ul,

    "Layers..." / Struct(
        "next_stage" / Int32ul,
        "stage_type" / Int32ul,
        "implementation" / Int32ul,
        "Op..." / Switch(this.stage_type,
            {
                # Convolution
                0: Struct(
                    "radixX" / Int32ul,
                    "radixY" / Int32ul,
                    "radixStrideX" / Int32ul,
                    "radixStrideY" / Int32ul,
                    "padX" / Int32ul,
                    "padY" / Int32ul,
                    "padStyle" / Int32ul,
                    "dilation" / Int32ul,
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[4]
                ),
                # MaxPooling
                1: Struct(
                    "radixX" / Int32ul,
                    "radixY" / Int32ul,
                    "radixStrideX" / Int32ul,
                    "radixStrideY" / Int32ul,
                    "padX" / Int32ul,
                    "padY" / Int32ul,
                    "padStyle" / Int32ul,
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[2]
                ),
                # AvePooling
                2: Struct(
                    "radixX" / Int32ul,
                    "radixY" / Int32ul,
                    "radixStrideX" / Int32ul,
                    "radixStrideY" / Int32ul,
                    "padX" / Int32ul,
                    "padY" / Int32ul,
                    "padStyle" / Int32ul,
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[2]
                ),
                # Softmax
                3: Struct(
                    "axis" / Int32ul,
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[2]
                ),
                # fully_connected_layer
                4: Struct(
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[4]
                ),
                # relu
                6: Struct(
                    "opX" / Int32ul,
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[2],
                    "postStrideX" / Int32ul,
                    "postStrideY" / Int32ul,
                ),
                # eltwise_sum
                12: Struct(
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[4]
                ),
                # eltwise_prod
                13: Struct(
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[4]
                ),
                # eltwise_max
                14: Struct(
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[4]
                ),

                # Hardware Convolution
                33: Struct(
                    "OpMode" / Int32ul,
                    "inputSize" / Int32ul,
                    "outputSize" / Int32ul,
                    "concatOffset" / Int32ul,
                    "unloadCMX" / Int32ul,
                    "overwriteInput" / Int32ul,
                    "CMXSize" / Int32ul,
                    "ReluSHVAccum" / Int32ul,
                    "ShvNegSlope" / Int32ul,
                    "ShvPosSlope" / Int32ul,
                    "DescrAmount" / Int32ul,
                    "Descriptors" / Struct(
                        "Half-Line" / Bytes(4)[32] # Int32ul[32]
                    )[this.DescrAmount],
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[5]
                ),
                # conversion
                37: Struct(
                    "Buffers..." / Struct(
                        "x" / Int32ul,
                        "y" / Int32ul,
                        "z" / Int32ul,
                        "x_S" / Int32ul,
                        "y_S" / Int32ul,
                        "z_S" / Int32ul,
                        "offset" / Int32ul,
                        "location" / Int32ul,
                        "datatype" / Int32ul,
                        "order" / Int32ul,
                    )[2]
                ),
            }
        ),
        "preop_type" / Int32ul,
        "postop_type" / Int32ul,
        "PostOp...." / Switch(this.postop_type,{
            # ReLU
            6: Struct(
                "opX" / Int32ul,
                "postStrideX" / Int32ul,
                "postStrideY" / Int32ul
            ),
            # # Bias
            # 9: Struct(

            # )

        })
    )[this.stage_count],

    # "paduntil" / RepeatUntil(lambda x, lst, ctx: x > 0, Int32ul),
    # "size_of_buffer_section" / Computed(this.paduntil[-1]),
    # "Post_Pad" / Padded(3, Byte),

    # "buffers" / Array(this.size_of_buffer_section - 7, Byte, discard=True),
    # "relocation_buffer_size" / Int32ul,
    # "blob_buffer_reloc_offset" / Int32ul,
    # "blob_buffer_reloc_size" / Int32ul,
    # "work_buffer_reloc_offset" / Int32ul,
    # "work_buffer_reloc_size" / Int32ul,

    # "blob_reloc_entries" /  Struct(
    #     "offset" / Int32ul,
    #     "location" / Int32ul,
    # )[this.blob_buffer_reloc_size // 8],


    # "work_reloc_entries" /  Struct(
    #     "offset" / Int32ul,
    #     "location" / Int32ul,
    # )[this.work_buffer_reloc_size // 8],

)

def main():
    parser = argparse.ArgumentParser(description="""A quick debug tool for Blob files.""",
                                        formatter_class=argparse.RawTextHelpFormatter, add_help=False)
    parser.add_argument('--file', metavar='', type=str, nargs='?',
                        help='path to blob')
    args = parser.parse_args()



    a = blob_format.parse_file(args.file)
    print(a)

    for i, s in enumerate(a["Layers..."]):
        try:
            for j, d in enumerate(s["Op..."]["Descriptors"]):
                descriptor = ""
                for x in d["Half-Line"]:
                    # print("original Hex", x, x.hex())
                    # descriptor += x.hex()
                    import struct
                    # val = struct.pack(">I", int(x.hex()))
                    # print(x.hex(), type(x.hex()))
                    val = x.hex()
                    # print("Original:  ", val)
                    # print("Hex: L:", val[:4], "R:", val[4:])
                    lval = val[:4]
                    rval = val[4:]
                    # print("LR: ", lval, rval)
                    lval = int(lval, 16)
                    rval = int(rval, 16)
                    # print("LR: ", lval, rval)
                    lval = struct.pack('<H',lval)
                    rval = struct.pack('<H',rval)
                    # print("LR: ", lval, rval)
                    # val = struct.unpack('>I', val)
                    # print("VAL", val)
                    # print(lval+rval)
                    val = rval+lval
                    # print("Converted: ", val.hex())
                    descriptor += val.hex()


                print("")
                print("Descriptor #", j)
                s = SerializedDescriptor("Conv")
                s.deserialize(descriptor)
                s.print()
                # quit()

        except Exception as e:
            # print(e)
            print("#### No Descriptors at index", i, "####")

if __name__ == "__main__":
    main()

