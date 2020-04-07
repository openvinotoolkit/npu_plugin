#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metamodel" for configuration "Release"
set_property(TARGET metamodel APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(metamodel PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmetamodel.so"
  IMPORTED_SONAME_RELEASE "libmetamodel.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS metamodel )
list(APPEND _IMPORT_CHECK_FILES_FOR_metamodel "${_IMPORT_PREFIX}/lib/libmetamodel.so" )

# Import target "cm" for configuration "Release"
set_property(TARGET cm APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cm PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcm.so"
  IMPORTED_SONAME_RELEASE "libcm.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS cm )
list(APPEND _IMPORT_CHECK_FILES_FOR_cm "${_IMPORT_PREFIX}/lib/libcm.so" )

# Import target "model" for configuration "Release"
set_property(TARGET model APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(model PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmodel.so"
  IMPORTED_SONAME_RELEASE "libmodel.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS model )
list(APPEND _IMPORT_CHECK_FILES_FOR_model "${_IMPORT_PREFIX}/lib/libmodel.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
