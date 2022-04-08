DATA_FOLDER ?= $(APPDIR)/../data/

test-defines-y                     += -DICV_TESTS_SUPPORT
test-defines-y                     += -D'DATA_FOLDER=${DATA_FOLDER}'
test-defines-$(CONFIG_ALL_TESTS)   += -DALL_TESTS
test-defines-$(CONFIG_SOFT_ASSERT) += -DSOFT_ASSERT

cppopt-los-y += $(test-defines-y)
cppopt-lrt-y += $(test-defines-y)

subdirs-los-y += ../../leon ../../../..
subdirs-lrt-y += ../../leon ../../../..
subdirs-lrt-$(CONFIG_TARGET_SOC_3720) +=  ../../../../act_shave_lib

#
# Run parameters
#

run-mode-value-y                              = 0
run-mode-value-$(CONFIG_TEST_MODE_LISTSUITES) = 1
run-mode-value-$(CONFIG_TEST_MODE_LISTTESTS)  = 2

print-name-value-y                            = 0
print-name-value-$(CONFIG_TEST_MODE_BRIEF)    = 1
print-name-value-$(CONFIG_TEST_MODE_FULL)     = 1

print-time-value-y                            = 0
print-time-value-$(CONFIG_TEST_MODE_BRIEF)    = 1
print-time-value-$(CONFIG_TEST_MODE_FULL)     = 1

print-params-value-y                          = 0
print-params-value-$(CONFIG_TEST_MODE_FULL)   = 1

check-result-value-y                          = 0
check-result-value-$(CONFIG_TEST_MODE_QUIET)  = 1
check-result-value-$(CONFIG_TEST_MODE_BRIEF)  = 2
check-result-value-$(CONFIG_TEST_MODE_FULL)   = 3

call-once-value-y                   = 0
call-once-value-$(CONFIG_CALL_ONCE) = 1

printf-diffs-value-y 				= 0
printf-diffs-value-$(CONFIG_PRINT_DIFFS) = 1

print-perf-counters-value-y = 0
print-perf-counters-value-$(CONFIG_PRINT_PERF_COUNTERS) = 1

#mvdbg-opt-y += --no-uart
mvdbg-opt-y += -D:MDK_ROOT_PATH=$(MDK_ROOT_PATH)
mvdbg-opt-y += -D:RUN_MODE=$(run-mode-value-y)
mvdbg-opt-y += -D:PRINT_NAME=$(print-name-value-y)
mvdbg-opt-y += -D:PRINT_TIME=$(print-time-value-y)
mvdbg-opt-y += -D:PRINT_PARAMS=$(print-params-value-y)
mvdbg-opt-y += -D:CHECK_RESULT=$(check-result-value-y)
mvdbg-opt-y += -D:CALL_ONCE=$(call-once-value-y)
mvdbg-opt-y += -D:PRINT_DIFFS=$(printf-diffs-value-y)
mvdbg-opt-y += -D:MIN_SHAVES=$(CONFIG_MIN_SHAVES)
mvdbg-opt-y += -D:MAX_SHAVES=$(CONFIG_MAX_SHAVES)
mvdbg-opt-y += -D:REPEAT_COUNT=$(CONFIG_REPEAT_COUNT)
mvdbg-opt-y += -D:PRINT_PERF_COUNTERS=$(print-perf-counters-value-y)
mvdbg-opt-y += -D:TEST_FILTER=$(call unquote,$(CONFIG_TEST_FILTER))
