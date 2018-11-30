## Dependencies
find_package( dlib )

set( INCLUDES ../report ../mc ../features ../data ../segmentation ${dlib_INCLUDE_DIRS})

set( LIBS ${MC_LIB}  )

add_definitions(-ftemplate-backtrace-limit=0)

### Generate the MC Pipeline executable
add_executable( MC_pipeline  MC_pipeline.cpp )
target_include_directories( MC_pipeline PRIVATE ${INCLUDES} ${CMAKE_SOURCE_DIR}/src/include )
target_link_libraries( MC_pipeline ${LIBS} stdc++fs )
target_compile_features( MC_pipeline PRIVATE cxx_std_17)
add_dependencies(MC_pipeline dlib fmt )

### Generate the feature scoring executable
#add_executable( markovian_features_scoring  markovian_features_analysis.cpp )
#target_include_directories( markovian_features_scoring PRIVATE ${INCLUDES} ${CMAKE_SOURCE_DIR}/src/include )
#target_link_libraries( markovian_features_scoring PRIVATE  ${LIBS}  stdc++fs  )
#target_compile_features( markovian_features_scoring PRIVATE cxx_std_17 )
#add_dependencies(markovian_features_scoring dlib fmt )

### Confusion Matrix Test
add_executable( segmentation segmentation.cpp )
target_include_directories( segmentation PRIVATE ${INCLUDES} ${CMAKE_SOURCE_DIR}/src/include )
target_link_libraries( segmentation PRIVATE  ${LIBS} ${SEGMENTATION_LIB}    )
target_compile_features( segmentation PRIVATE cxx_std_17)
add_dependencies(segmentation dlib fmt )


### Segmentation
add_executable( confusionMatrix confusionMatrix.cpp )
target_include_directories( confusionMatrix PRIVATE ${INCLUDES} ${CMAKE_SOURCE_DIR}/src/include )
target_link_libraries( confusionMatrix PRIVATE  ${LIBS}    )
target_compile_features( confusionMatrix PRIVATE cxx_std_17)
add_dependencies(confusionMatrix dlib fmt )

# Install the executable (generate with 'make install' in the bin directory)
#install( TARGETS ${FINDING_CLUMPS_EXE} DESTINATION bin )
