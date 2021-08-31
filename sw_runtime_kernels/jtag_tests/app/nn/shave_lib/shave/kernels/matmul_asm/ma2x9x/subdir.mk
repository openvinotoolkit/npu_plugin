#Using += allows for secondary expansion which renders this construct incorrect
srcs-shave-y += $(wildcard ./*.asm)

$(eval $(call mdk-redirect-srcs,srcs-shave-y,shavelib-objs-y))
