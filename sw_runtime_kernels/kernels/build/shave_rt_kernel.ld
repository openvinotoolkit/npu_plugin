OUTPUT_FORMAT("elf32-sparc")
OUTPUT_ARCH(sparc)

SECTIONS
{  
    . = 0x1c000000;      
    .text : {               
        *(.text*)
        . = ALIGN(16);
        *(.rodata*)
        . = ALIGN(16);
        KEEP(*(.uuid.rodata*))           
        *(.uuid.rodata*)
    } 
}
