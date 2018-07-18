import construct
from construct import *
from construct import Struct
import ctypes
import argparse

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
                    )[3]
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
                    )[3]
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
                    )[3]
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
            )
        })
    )[this.stage_count],

    "paduntil" / RepeatUntil(lambda x, lst, ctx: x > 0, Int32ul),
    "size_of_buffer_section" / Computed(this.paduntil[-1]),
    "Post_Pad" / Padded(3, Byte),

    "buffers" / Array(this.size_of_buffer_section - 7, Byte, discard=True),
    "relocation_buffer_size" / Int32ul,
    "blob_buffer_reloc_offset" / Int32ul,
    "blob_buffer_reloc_size" / Int32ul,
    "work_buffer_reloc_offset" / Int32ul,
    "work_buffer_reloc_size" / Int32ul,

    "blob_reloc_entries" /  Struct(
        "offset" / Int32ul,
        "location" / Int32ul,
    )[this.blob_buffer_reloc_size // 8],


    "work_reloc_entries" /  Struct(
        "offset" / Int32ul,
        "location" / Int32ul,
    )[this.work_buffer_reloc_size // 8],

)

def main():
    parser = argparse.ArgumentParser(description="""A quick debug tool for Blob files.""",
                                        formatter_class=argparse.RawTextHelpFormatter, add_help=False)
    parser.add_argument('--file', metavar='', type=str, nargs='?',
                        help='path to blob')
    args = parser.parse_args()



    a = blob_format.parse_file(args.file)
    print(a)
    assert a["e_type"] == 1
    assert a["e_machine"] == 2
    assert a["e_version"] == 1
    assert a["e_ehsize"] == 272
    assert a["magic_blob"] == 8708
    assert 1000 > a["stage_count"] > 0


if __name__ == "__main__":
    main()