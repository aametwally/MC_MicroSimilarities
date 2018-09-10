
#include "ZYPipeline.hpp"
#include "clara.hpp"


std::vector<std::string> splitParameters( std::string params )
{
    io::trim( params, "[({})]" );
    return io::split( params, "," );
}

int main( int argc, char *argv[] )
{
    using io::join;
    std::string input;
    std::string fastaFormat = keys( FormatLabels ).front();
    std::string order = "3";
    size_t k = 10;
    bool showHelp = false;
    std::string grouping = keys( GroupingLabels ).front();

    auto cli
            = clara::Arg( input, "input" )
                      ( "UniRef input file" )
              | clara::Opt( grouping, join( keys( GroupingLabels ), "|" ))
              ["-G"]["--grouping"]
                      ( fmt::format( "grouping method, default:{}", grouping ))
              | clara::Opt( fastaFormat, join( keys( FormatLabels ), "|" ))
              ["-f"]["--fformat"]
                      ( fmt::format( "input file processor, default:{}", fastaFormat ))
              | clara::Opt( order , "Markov chain order" )
              ["-o"]["--order"]
                      ( fmt::format( "default:{}", order ))
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
        fmt::print( "[Args][input:{}]"
                    "[fformat:{}]"
                    "[order:{}-{}]"
                    "[k-fold:{}]"
                    "[grouping:{}]\n",
                    input, fastaFormat, order,
                    k, grouping );

        for (auto &order : splitParameters( order ))
        {
            for (auto &g : splitParameters( grouping ))
            {

                auto o = std::stoi( order );
                fmt::print( "[Params]"
                            "[order:{}]"
                            "[grouping:{}]\n", o, g );

                std::visit( [&]( auto &&p ) {
                    p.runPipeline_VALIDATION( LabeledEntry::loadEntries( input, fastaFormat ), o , k );
                }, getZYPipeline( g ));
            }
        }
    }
    return 0;
}
