include-dirs-shave_nn-y += inc
srcs-shave-$(CONFIG_TARGET_SOC_MA2490) += $(wildcard ./2490/*.c*) $(wildcard ./*.asm)
srcs-shave_nn-$(CONFIG_TARGET_SOC_3720) += $(wildcard ./*.c*) $(wildcard ./*.asm)
#srcs-shave_nn-y += $(wildcard src/*.c*) $(wildcard src/*.asm)
shavelib-preserved-symbols-y += preSingleSoftmax 
