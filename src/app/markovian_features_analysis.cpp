//
// Created by asem on 19/08/18.
//

//
// Created by asem on 13/08/18.
//


#include "FeatureScoringPipeline.hpp"
#include "clara.hpp"


#include <iomanip>

std::string timeNow()
{
    auto n = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t( n );
    auto bf = std::localtime( &in_time_t );

    std::stringstream ss;
    ss << std::put_time( bf, "%Y-%m-%d %X" );

    return ss.str();
}


std::vector<std::string> splitParameters( std::string params )
{
    io::trim( params, "[({})]" );
    return io::split( params, "," );
}

std::string prefix( const std::string &input,
                    const std::string &o,
                    const std::string &g )
{
    return fmt::format( "[{}][{}][order:{}][grouping:{}]", timeNow(), input, o, g );
}

int main( int argc, char *argv[] )
{
    using io::join;
    std::string input, testFile;
    std::string fastaFormat = keys( FormatLabels ).front();
    std::string markovianOrder = "3";
    size_t k = 10;
    bool showHelp = false;
    std::string grouping = keys( GroupingLabels ).front();

    auto cli
            = clara::Arg( input, "input" )
                      ( "UniRef input file" )
              | clara::Opt( testFile, "test" )
              ["-T"]["--test"]
                      ( "test file" )
              | clara::Opt( grouping, join( keys( GroupingLabels ), "|" ))
              ["-G"]["--grouping"]
                      ( fmt::format( "grouping method, default:{}", grouping ))
              | clara::Opt( fastaFormat, join( keys( FormatLabels ), "|" ))
              ["-f"]["--fformat"]
                      ( fmt::format( "input file processor, default:{}", fastaFormat ))
              | clara::Opt( markovianOrder, "order" )
              ["-o"]["--order"]
                      ( fmt::format( "Markovian order, default:{}", markovianOrder ))
              | clara::Opt( k, "k-fold" )
              ["-k"]["--k-fold"]
                      ( fmt::format( "cross validation k-fold, default:{}", k ))
              | clara::Help( showHelp );

    auto result = cli.parse( clara::Args( argc, argv ));
    if ( !result )
    {
        fmt::print( "Error in command line:{}\n", result.errorMessage());
        exit( 1 );
    } else if ( showHelp )
    {
        cli.writeToStream( std::cout );
    } else if ( !testFile.empty())
    {
        fmt::print( "[Args][input:{}][testFile:{}][order:{}][kfold:{}]\n",
                    input, testFile, markovianOrder, k );


    } else
    {
        fmt::print( "[Args][input:{}]"
                    "[fformat:{}]"
                    "[order:{}]"
                    "[k-fold:{}]"
                    "[grouping:{}]\n",
                    input, fastaFormat,
                    markovianOrder,
                    k, grouping );
        namespace fs = std::experimental::filesystem;
        fs::path p( input.c_str());
        const std::string fname = p.filename();

        for (auto &g : splitParameters( grouping ))
            for (auto &o : splitParameters( markovianOrder ))
            {
                fmt::print( "[Params]"
                            "[order:{}]"
                            "[grouping:{}]\n", o, g );
                std::visit( [&]( auto &&p ) {
                    p.runPipeline_VALIDATION( LabeledEntry::loadEntries( input, fastaFormat ),
                                              std::stoi( o ), k, prefix( fname, o, g ));
                }, getFeatureScoringPipeline( g ));
            }


    }
    return 0;
}
