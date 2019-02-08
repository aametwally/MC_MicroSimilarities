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

std::string prefix(
        const std::string &input,
        MC::Order mnOrder,
        MC::Order mxOrder,
        const std::string &g
)
{
    return fmt::format( "[{}][{}][order:{}-{}][grouping:{}]", timeNow(), input, mnOrder, mxOrder, g );
}

int main(
        int argc,
        char *argv[]
)
{
    using namespace MC;
    using io::join;
    std::string input, testFile;
    std::string fastaFormat = keys( FormatLabels ).front();
    std::string minOrder = "3";
    std::string maxOrder = "5";
    std::string order = "";
    size_t k = 10;
    bool showHelp = false;
    std::string grouping = keys( GroupingLabels ).front();
    std::string model = keys( MCModelLabels ).front();

    auto cli
            = clara::Arg( input, "input" )
                      ( "UniRef input file" )
              | clara::Opt( testFile, "test" )
              ["-T"]["--test"]
                      ( "test file" )
              | clara::Opt( model, join( keys( MCModelLabels ), "|" ))
              ["-m"]["--model"]
                      ( fmt::format( "Markov Chains model, default:{}", model ))
              | clara::Opt( grouping, join( keys( GroupingLabels ), "|" ))
              ["-G"]["--grouping"]
                      ( fmt::format( "grouping method, default:{}", grouping ))
              | clara::Opt( fastaFormat, join( keys( FormatLabels ), "|" ))
              ["-f"]["--fformat"]
                      ( fmt::format( "input file processor, default:{}", fastaFormat ))
              | clara::Opt( order, "MC order" )
              ["-o"]["--order"]
                      ( fmt::format( "Specify MC of higher order o, default:{}", maxOrder ))
              | clara::Opt( minOrder, "minimum order" )
              ["-l"]["--min-order"]
                      ( fmt::format( "Markovian lower order, default:{}", minOrder ))
              | clara::Opt( maxOrder, "maximum order" )
              ["-h"]["--max-order"]
                      ( fmt::format( "Markovian higher order, default:{}", maxOrder ))
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
    } else
    {
        if ( !order.empty())
            maxOrder = order;

        fmt::print( "[Args][input:{}]"
                    "[fformat:{}]"
                    "[model:{}]"
                    "[order:{}-{}]"
                    "[k-fold:{}]"
                    "[grouping:{}]\n",
                    input, fastaFormat,
                    model,
                    minOrder, maxOrder,
                    k, grouping );
        namespace fs = std::experimental::filesystem;
        fs::path p( input.c_str());
        const std::string fname = p.filename();


        for (auto &m : splitParameters( model ))
        {
            std::string _minOrder;
            if ( m == "romc" )
                _minOrder = minOrder;
            else _minOrder = "1";
            for (auto &min : splitParameters( _minOrder ))
                for (auto &max : splitParameters( maxOrder ))
                {
                    auto mnOrder = std::stoi( min );
                    auto mxOrder = std::stoi( max );
                    if ( mxOrder >= mnOrder )
                    {
                        for (auto &g : splitParameters( grouping ))
                        {
                            fmt::print( "[Params]"
                                        "[model:{}]"
                                        "[order:{}-{}]"
                                        "[grouping:{}]\n", m, mnOrder, mxOrder, g );
                            std::visit( [&]( auto &&p ) {
                                p.runPipeline_VALIDATION( LabeledEntry::loadEntries( input, fastaFormat ),
                                                          k, prefix( fname, mnOrder, mxOrder, g ));
                            }, getFeatureScoringPipeline( g, m, mnOrder, mxOrder ));
                        }
                    }

                }
        }

    }
    return 0;
}
