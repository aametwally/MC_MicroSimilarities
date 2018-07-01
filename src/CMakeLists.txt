### Library name: lasso
set( MARKOVIAN_FEATURES_LIB "markovian_features" )


### Library sources/headers
set( MARKOVIAN_FEATURES_LIB_SOURCE_FILES  )

set( MARKOVIAN_FEATURES_LIB_HEADER_FILES  )


### Include directories.
include_directories( . )
include_directories( algorithm )



#### Define the library
#add_library( ${MARKOVIAN_FEATURES_LIB}
#    ${MARKOVIAN_FEATURES_LIB_SOURCE_FILES}
#    ${MARKOVIAN_FEATURES_LIB_HEADER_FILES})

#set_target_properties( ${MARKOVIAN_FEATURES_LIB} PROPERTIES LINKER_LANGUAGE CXX COMPILE_DEFINITIONS "LIBRARY_API=1")

#install( TARGETS ${MARKOVIAN_FEATURES_LIB} DESTINATION lib )
# install( FILES ${MARKOVIAN_FEATURES_LIB_HEADER_FILES} DESTINATION include/${MARKOVIAN_FEATURES_LIB} )


# Includes Catch in the project:
add_subdirectory( app )





