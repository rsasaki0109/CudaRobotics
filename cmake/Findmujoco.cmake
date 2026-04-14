include(FindPackageHandleStandardArgs)

set(_mujoco_roots "")

if(DEFINED mujoco_ROOT)
  list(APPEND _mujoco_roots "${mujoco_ROOT}")
endif()
if(DEFINED MUJOCO_ROOT)
  list(APPEND _mujoco_roots "${MUJOCO_ROOT}")
endif()
if(DEFINED ENV{MUJOCO_ROOT})
  list(APPEND _mujoco_roots "$ENV{MUJOCO_ROOT}")
endif()
if(DEFINED ENV{MUJOCO_HOME})
  list(APPEND _mujoco_roots "$ENV{MUJOCO_HOME}")
endif()

file(GLOB _mujoco_venv_candidates
  "${CMAKE_SOURCE_DIR}/.venv-mujoco/lib/python*/site-packages/mujoco"
  "${PROJECT_SOURCE_DIR}/.venv-mujoco/lib/python*/site-packages/mujoco"
)
list(APPEND _mujoco_roots ${_mujoco_venv_candidates})

if(DEFINED ENV{VIRTUAL_ENV})
  file(GLOB _mujoco_virtualenv_candidates
    "$ENV{VIRTUAL_ENV}/lib/python*/site-packages/mujoco"
  )
  list(APPEND _mujoco_roots ${_mujoco_virtualenv_candidates})
endif()

list(APPEND _mujoco_roots
  "/usr"
  "/usr/local"
)

list(REMOVE_DUPLICATES _mujoco_roots)

find_path(mujoco_INCLUDE_DIR
  NAMES mujoco/mujoco.h
  PATHS ${_mujoco_roots}
  PATH_SUFFIXES include
)

find_library(mujoco_LIBRARY
  NAMES mujoco
  PATHS ${_mujoco_roots}
  PATH_SUFFIXES lib
)

if(NOT mujoco_LIBRARY)
  foreach(_root IN LISTS _mujoco_roots)
    file(GLOB _mujoco_globbed_libs
      "${_root}/libmujoco.so*"
      "${_root}/lib/libmujoco.so*"
    )
    if(_mujoco_globbed_libs)
      list(GET _mujoco_globbed_libs 0 mujoco_LIBRARY)
      break()
    endif()
  endforeach()
endif()

find_package_handle_standard_args(mujoco
  REQUIRED_VARS mujoco_INCLUDE_DIR mujoco_LIBRARY
)

if(mujoco_FOUND)
  get_filename_component(mujoco_LIBRARY_DIR "${mujoco_LIBRARY}" DIRECTORY)
  if(NOT TARGET mujoco::mujoco)
    add_library(mujoco::mujoco SHARED IMPORTED)
    set_target_properties(mujoco::mujoco PROPERTIES
      IMPORTED_LOCATION "${mujoco_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${mujoco_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(mujoco_INCLUDE_DIR mujoco_LIBRARY mujoco_LIBRARY_DIR)
