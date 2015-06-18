###############################################################################
# Find LibE57
#
# This sets the following variables:
# LIBE57_FOUND - True if LibE57 was found.
# LIBE57_INCLUDE_DIRS - Directories containing the LibE57 include files.
# LIBE57_LIBRARY_DIRS - Directories containing the LibE57 library.
# LIBE57_LIBRARIES - LibE57 library files.

find_path(LIBE57_INCLUDE_DIR e57
    HINTS "/usr/include" "/usr/local/include" "$ENV{PROGRAMFILES}")
#HINTS "/usr/include" "/usr/local/include" "/usr/x86_64-w64-mingw32/include" "$ENV{PROGRAMFILES}")

if (WIN32)
    find_library(LIBE57_LIBRARY_PATH E57RefImpl HINTS "/usr/x86_64-w64-mingw32/lib")
    find_library(LIBE57SIMPLE_LIBRARY_PATH E57SimpleImpl HINTS "/usr/x86_64-w64-mingw32/lib")
    find_library(TIME_CONVERSION_LIBRARY_PATH time_conversion HINTS "/usr/x86_64-w64-mingw32/lib")
    find_library(XERCES_LIBRARY_PATH xerces-c HINTS "/usr/x86_64-w64-mingw32/lib")
else()
    find_library(LIBE57_LIBRARY_PATH E57RefImpl HINTS "/usr/lib" "/usr/local/lib")
    find_library(LIBE57SIMPLE_LIBRARY_PATH E57SimpleImpl HINTS "/usr/lib" "/usr/local/lib")
    find_library(TIME_CONVERSION_LIBRARY_PATH time_conversion HINTS "/usr/lib" "/usr/local/lib")
    find_library(XERCES_LIBRARY_PATH xerces-c HINTS "/usr/lib" "/usr/local/lib")
endif()

if(EXISTS ${LIBE57_LIBRARY_PATH})
    get_filename_component(LIBE57_LIBRARY ${LIBE57_LIBRARY_PATH} NAME)
    if (WIN32)
        find_path(LIBE57_LIBRARY_DIR ${LIBE57_LIBRARY} HINTS "/usr/lib" "/usr/local/lib" "/usr/x86_64-w64-mingw32/lib/")
    else()
        find_path(LIBE57_LIBRARY_DIR ${LIBE57_LIBRARY} HINTS "/usr/lib" "/usr/local/lib")
    endif()
endif()

if(EXISTS ${LIBE57SIMPLE_LIBRARY_PATH})
    get_filename_component(LIBE57SIMPLE_LIBRARY ${LIBE57SIMPLE_LIBRARY_PATH} NAME)
    if (WIN32)
        find_path(LIBE57SIMPLE_LIBRARY_DIR ${LIBE57SIMPLE_LIBRARY} HINTS "/usr/lib" "/usr/local/lib" "/usr/x86_64-w64-mingw32/lib/")
    else()
        find_path(LIBE57SIMPLE_LIBRARY_DIR ${LIBE57SIMPLE_LIBRARY} HINTS "/usr/lib" "/usr/local/lib")
    endif()
endif()

if(EXISTS ${TIME_CONVERSION_LIBRARY_PATH})
    get_filename_component(TIME_CONVERSION_LIBRARY ${TIME_CONVERSION_LIBRARY_PATH} NAME)
    if (WIN32)
        find_path(TIME_CONVERSION_LIBRARY_DIR ${TIME_CONVERSION_LIBRARY} HINTS "/usr/lib" "/usr/local/lib" "/usr/x86_64-w64-mingw32/lib/")
    else()
        find_path(TIME_CONVERSION_LIBRARY_DIR ${TIME_CONVERSION_LIBRARY} HINTS "/usr/lib" "/usr/local/lib")
    endif()
endif()

if(EXISTS ${XERCES_LIBRARY_PATH})
    get_filename_component(XERCES_LIBRARY ${XERCES_LIBRARY_PATH} NAME)
    if (WIN32)
        find_path(XERCES_LIBRARY_DIR ${XERCES_LIBRARY} HINTS "/usr/lib" "/usr/local/lib" "/usr/x86_64-w64-mingw32/lib/")
    else()
        find_path(XERCES_LIBRARY_DIR ${XERCES_LIBRARY} HINTS "/usr/lib" "/usr/local/lib")
    endif()
endif()

set(LIBE57_INCLUDE_DIRS ${LIBE57_INCLUDE_DIR})
set(LIBE57_LIBRARY_DIRS ${LIBE57_LIBRARY_DIR})
set(LIBE57_LIBRARIES ${LIBE57_LIBRARY} ${LIBE57SIMPLE_LIBRARY} ${TIME_CONVERSION_LIBRARY} ${XERCES_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibE57 DEFAULT_MSG LIBE57_INCLUDE_DIR LIBE57_LIBRARY LIBE57_LIBRARY_DIR)

mark_as_advanced(LIBE57_INCLUDE_DIR)
mark_as_advanced(LIBE57_LIBRARY_DIR)
mark_as_advanced(LIBE57_LIBRARY)
mark_as_advanced(LIBE57_LIBRARY_PATH)
