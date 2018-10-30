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
target_include_directories( ${REPORT_LIB} INTERFACE . ${CMAKE_SOURCE_DIR}/src/include  )
target_link_libraries( ${REPORT_LIB} INTERFACE fmt-header-only stdc++fs )
target_compile_features( ${REPORT_LIB} INTERFACE cxx_std_17)