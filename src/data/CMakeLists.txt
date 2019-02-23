

file( STRINGS "aaindex1" __AAINDEX1_DATA )
string( REPLACE "\"" "\\\"" _AAINDEX1_DATA "${__AAINDEX1_DATA}" )
foreach( line IN LISTS _AAINDEX1_DATA )
  string( APPEND AAINDEX1_DATA "${line}\\n\\\n" )
endforeach()


configure_file( AAIndex1Data.h.in ${CMAKE_CURRENT_SOURCE_DIR}/AAIndex1Data.h @ONLY )

install(
  DIRECTORY   mcms
  DESTINATION include
  COMPONENT   library
  FILES_MATCHING PATTERN "*.{hpp,h}"
)