set( REPORT_INCLUDES
        Timers.hpp
        ConfusionMatrix.hpp
        CrossValidationStatistics.hpp
        EnsembleCrossValidation.hpp
        crossvalidation.hpp
        FeatureScoreAUC.hpp )

set( REPORT_SOURCES
        ConfusionMatrix.cpp
        CrossValidationStatistics.cpp
        EnsembleCrossValidation.cpp )

set( REPORT_LIB report_lib )

add_library( ${REPORT_LIB} STATIC  ${REPORT_INCLUDES} ${REPORT_SOURCES} )
add_dependencies(${REPORT_LIB}  fmt )

target_include_directories( ${REPORT_LIB} PUBLIC .
        ${CMAKE_SOURCE_DIR}/src/include
        ${EXTERNAL_INSTALL_LOCATION}/include
        ${CMAKE_SOURCE_DIR}/src/data )

target_link_libraries( ${REPORT_LIB} INTERFACE fmt-header-only stdc++fs )
target_compile_features( ${REPORT_LIB} INTERFACE cxx_std_17)