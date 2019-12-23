LINK_DIRECTORIES($ENV{MCM_HOME}/build/lib)
INCLUDE_DIRECTORIES($ENV{MCM_HOME})

add_executable(three_layer_conv_model three_layer_conv_model.cpp)
target_link_libraries(three_layer_conv_model cm)


add_custom_target(generate_lp_scheduler_blob)
add_custom_command(TARGET generate_lp_scheduler_blob
                   COMMAND ./three_layer_conv_model -d
                    ${CMAKE_SOURCE_DIR}/comp_desc_lp_scheduler.json
                   COMMAND rm -rf ./output)
add_dependencies(generate_lp_scheduler_blob three_layer_conv_model)


add_custom_target(generate_default_blob)
add_custom_command(TARGET generate_default_blob 
                   COMMAND ./three_layer_conv_model -d
                    ${CMAKE_SOURCE_DIR}/comp_desc_default.json
                   COMMAND rm -rf ./output)
add_dependencies(generate_default_blob three_layer_conv_model)


add_executable(generate_random_binary_streams
      generate_random_binary_streams.cpp)

add_custom_target(generate_test_bench)
add_custom_command(TARGET generate_test_bench
                   COMMAND echo "1x56x56x3" > in.txt
                   COMMAND ./generate_random_binary_streams -n 4 < in.txt
                   COMMAND rm in.txt)
add_dependencies(generate_test_bench generate_random_binary_streams
                 generate_default_blob generate_lp_scheduler_blob)
