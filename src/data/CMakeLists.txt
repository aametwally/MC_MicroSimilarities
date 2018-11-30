

file(STRINGS "aaindex1" __AAINDEX1_DATA )
string(REPLACE "\"" "\\\"" _AAINDEX1_DATA "${__AAINDEX1_DATA}")
foreach(line IN LISTS _AAINDEX1_DATA)
    string(APPEND AAINDEX1_DATA "${line}\\\n")
endforeach()


configure_file( AAIndex1Data.h.in ${CMAKE_CURRENT_SOURCE_DIR}/AAIndex1Data.h @ONLY)

#add_library( DATA_LIB ${DATA_INCLUDES} ${DATA_SOURCES} )
#target_include_directories( DATA_LIB PRIVATE ${CMAKE_SOURCE_DIR}/src/include  )
#target_link_libraries( DATA_LIB PRIVATE fmt-header-only stdc++fs )
#target_compile_features( DATA_LIB PRIVATE cxx_std_17)