
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was NCCLConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)

find_dependency(CUDAToolkit)
find_dependency(Threads)

include("${CMAKE_CURRENT_LIST_DIR}/NCCLTargets.cmake")

if(TARGET CUDA::cudart)
  get_target_property(_nccl_cuda_includes CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)
  if(_nccl_cuda_includes)
    foreach(_t NCCL::nccl NCCL::nccl_static)
      if(TARGET ${_t})
        set_property(TARGET ${_t} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${_nccl_cuda_includes}")
      endif()
    endforeach()
  endif()
  unset(_nccl_cuda_includes)
endif()

set(NCCL_VERSION "2.29.7")
set(NCCL_INCLUDE_DIRS "${PACKAGE_PREFIX_DIR}/include")
set(NCCL_LIBRARIES NCCL::nccl)

