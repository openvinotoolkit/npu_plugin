# {% copyright %}

set(TARGET_NAME elfUnitTests)

file(GLOB_RECURSE UNIT_SOURCES CONFIGURE_DEPENDS "*.cpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${UNIT_SOURCES})

addIeTargetTest(
    NAME ${TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    LINK_LIBRARIES
        elf
        IE::commonTestUtils
    LABELS
        elf
)
