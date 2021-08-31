#Using += allows for secondary expansion which renders this construct incorrect
subdirs-shave-y += ma2x9x
srcs-shave-y += $(wildcard ./*.cpp)

$(eval $(call mdk-redirect-srcs,srcs-shave-y,shavelib-objs-y))
