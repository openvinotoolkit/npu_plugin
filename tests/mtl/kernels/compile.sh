set -e
set -o pipefail

# Locate tools directory
MV_TOOLS_VERSION="21.06.3-internal"
MV_TOOLS_DIR="$HOME/WORK/Tools/Releases/General/$MV_TOOLS_VERSION/linux64"

MOVICOMPILE="$MV_TOOLS_DIR/bin/moviCompile"
MOVILLD="$MV_TOOLS_DIR/bin/moviLLD"
OBJCOPY="$MV_TOOLS_DIR/sparc-myriad-rtems-9.2.0/bin/sparc-myriad-rtems-objcopy"
READELF="$MV_TOOLS_DIR/sparc-myriad-rtems-9.2.0/bin/sparc-myriad-rtems-readelf"

# first argument is the kernel source file
INPUT_FILE=$1

# second argument is optional and specifies the build directory
OUTPUT_DIR=$2
if test -z "$OUTPUT_DIR"
then
   OUTPUT_DIR="$(dirname $0)/output"
fi
mkdir -p $OUTPUT_DIR

INPUT_BASENAME=$(basename $INPUT_FILE)
OUTPUT_BASENAME="$OUTPUT_DIR/${INPUT_BASENAME%.c}"
OBJ_FILE="$OUTPUT_BASENAME.o"
ASM_FILE="$OUTPUT_BASENAME.asm"
ELF_FILE="$OUTPUT_BASENAME.elf"
TEXT_FILE="$OUTPUT_BASENAME.text"
DATA_FILE="$OUTPUT_BASENAME.data"

LDSCRIPT_FILE="$(dirname $0)/kernel.ld"
LIB_DIR="$(dirname $0)/lib/30xxxx-leon"

# generate assembly output, for debugging purposes
$MOVICOMPILE -mcpu=3720xx -S $INPUT_FILE -o $ASM_FILE

# compile the kernel source file
$MOVICOMPILE -mcpu=3720xx -c $INPUT_FILE -o $OBJ_FILE

# link the object file into an ELF executable
$MOVILLD \
    -flavor gnu \
    -zmax-page-size=16 \
    --script $LDSCRIPT_FILE \
    -entry kernel_entry \
    --gc-sections \
    --strip-debug \
    --discard-all \
    $OBJ_FILE \
    "$LIB_DIR/mlibm.a" \
    "$LIB_DIR/mlibcrt.a" \
    --output $ELF_FILE

# extract the text, arg, and data sections
$OBJCOPY -O binary --only-section=.text $ELF_FILE $TEXT_FILE
$OBJCOPY -O binary --only-section=.arg.data $ELF_FILE "$ELF_FILE.arg"
$OBJCOPY -O binary --only-section=.data $ELF_FILE "$ELF_FILE.data"

# concatenate the arg & data into a single writable-data blob
cat "$ELF_FILE.arg" "$ELF_FILE.data" > $DATA_FILE

# for debugging convenience, print the contents of the elf container
$READELF -S $ELF_FILE > "${OUTPUT_BASENAME}_sections.txt"
$READELF -r $ELF_FILE > "${OUTPUT_BASENAME}_relocations.txt"
$READELF -s $ELF_FILE > "${OUTPUT_BASENAME}_symbols.txt"

