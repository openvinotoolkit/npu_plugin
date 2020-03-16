#!/bin/bash

clc_root=${VPU_CLC_PATH:-"$HOME/movi-cltools-19.11.2/CLTools/bin"}
clc_use_brebuilt=${VPU_CLC_USE_PREBUILT:-1}
elf_folder="elf"
# gen OpenCL files
if [ ! -d "${elf_folder}" ]; then
    mkdir -p ${elf_folder}
fi
for src in ./kernels/*.cl
do
    echo "compiling " $src
    filename=${src##*/}
    if [ ${clc_use_brebuilt} -eq 1 ]; then
        SHAVE_LDSCRIPT_DIR=$clc_root/../ldscripts-kmb \
        SHAVE_MYRIAD_LD_DIR=$clc_root             \
        SHAVE_MA2X8XLIBS_DIR=$clc_root/../lib     \
        $clc_root/clc --strip-binary-header  $src -o "./${elf_folder}/${filename%.cl}.elf"
    else
        $clc_root/clc --strip-binary-header $src -o "./${elf_folder}/${filename%.cl}.elf"
    fi
    rc=$?
    if [ ${rc} -ne 0 ]; then
        exit ${rc}
    fi
done
