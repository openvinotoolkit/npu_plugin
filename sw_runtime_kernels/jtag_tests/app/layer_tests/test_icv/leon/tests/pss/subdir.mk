include-dirs-los-y += .
include-dirs-lrt-y += .
srcs-los-y += $(wildcard *.cpp)
srcs-lrt-y += $(wildcard *.cpp)

# compile MvTensor testing support code (and do not compile upon firmware build)
ccopt-los-y += -DICV_TESTS_SUPPORT
ccopt-lrt-y += -DICV_TESTS_SUPPORT
