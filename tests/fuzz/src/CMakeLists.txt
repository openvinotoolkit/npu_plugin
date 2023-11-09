function(add_fuzz_test TEST_NAME TEST_SOURCES)
    add_executable(${TEST_NAME} ${TEST_SOURCES})
    target_link_libraries(${TEST_NAME} PUBLIC npu_mlir_compiler_static)
    set_target_properties(${TEST_NAME} PROPERTIES LINK_FLAGS "-fsanitize=fuzzer")
endfunction(add_fuzz_test)

add_subdirectory(pipeline_default_hw_mode)
