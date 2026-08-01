#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "NCCL::nccl" for configuration "Release"
set_property(TARGET NCCL::nccl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(NCCL::nccl PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/nccl.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/nccl.dll"
  )

list(APPEND _cmake_import_check_targets NCCL::nccl )
list(APPEND _cmake_import_check_files_for_NCCL::nccl "${_IMPORT_PREFIX}/lib/nccl.lib" "${_IMPORT_PREFIX}/bin/nccl.dll" )

# Import target "NCCL::nccl_static" for configuration "Release"
set_property(TARGET NCCL::nccl_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(NCCL::nccl_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/nccl_static.lib"
  )

list(APPEND _cmake_import_check_targets NCCL::nccl_static )
list(APPEND _cmake_import_check_files_for_NCCL::nccl_static "${_IMPORT_PREFIX}/lib/nccl_static.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
