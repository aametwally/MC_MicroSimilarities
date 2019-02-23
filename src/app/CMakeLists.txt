set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/app )

set( INCLUDES ../report ../mc ../features ../data ../segmentation )

set( LIBS ${MC_LIB} )

add_definitions( -ftemplate-backtrace-limit=0 )

### Generate the MC Pipeline executable
add_executable( MC_pipeline MC_pipeline.cpp )
target_include_directories( MC_pipeline PRIVATE ${INCLUDES} ${CMAKE_SOURCE_DIR}/src/include )
target_link_libraries( MC_pipeline ${LIBS} stdc++fs ${CONAN_LIBS} )
target_compile_features( MC_pipeline PRIVATE cxx_std_17 )

### Generate the feature scoring executable
#add_executable( markovian_features_scoring  markovian_features_analysis.cpp )
#target_include_directories( markovian_features_scoring PRIVATE ${INCLUDES} ${CMAKE_SOURCE_DIR}/src/include )
#target_link_libraries( markovian_features_scoring PRIVATE  ${LIBS}  stdc++fs  )
#target_compile_features( markovian_features_scoring PRIVATE cxx_std_17 )
#add_dependencies(markovian_features_scoring dlib fmt )


### Segmentation
#add_executable( confusionMatrix confusionMatrix.cpp )
#target_include_directories( confusionMatrix PRIVATE ${INCLUDES} ${CMAKE_SOURCE_DIR}/src/include )
#target_link_libraries( confusionMatrix PRIVATE ${LIBS} ${CONAN_LIBS})
#target_compile_features( confusionMatrix PRIVATE cxx_std_17 )


### Protein Time Series Generator
add_executable( protein2ts protein2ts.cpp )
target_include_directories( protein2ts PRIVATE ${INCLUDES} ${CMAKE_SOURCE_DIR}/src/include )
target_link_libraries( protein2ts PRIVATE ${LIBS} ${CONAN_LIBS} )
target_compile_features( protein2ts PRIVATE cxx_std_17 )

# Installation
install( TARGETS MC_pipeline protein2ts
  RUNTIME DESTINATION bin )