OUTPUT_FORMAT("elf32-sparc")
OUTPUT_ARCH(sparc)

SECTIONS
{  
    . = 0x1c000000;      
    .text : {        
        *(.text* .rodata*)
    }
    
    . = 0x1e000000;
    
    .arg.data : { 
        KEEP(*(.arg.data))
        *(.arg.data) 
    }

    .data : { 
        *(.data*) 
    }
}