#include "markovian_features.hpp"
#include "clara.hpp"


std::vector<std::string> splitParameters( std::string params )
{
    io::trim( params, "[({})]" );
    return io::split( params, "," );
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
    std::string criteria = keys( CriteriaLabels ).front();
    std::string strategy = keys( ClassificationStrategyLabel ).front();

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
              | clara::Opt( criteria, join( keys( CriteriaLabels ), "|" ))
              ["-c"]["--criteria"]
                      ( fmt::format( "Similarity Criteria, default:{}", criteria ))
              | clara::Opt( strategy, join( keys( ClassificationStrategyLabel ), "|" ))
              ["-s"]["--strategy"]
                      ( fmt::format( "Classification Strategy, default:{}", strategy ))
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
                    "[criteria:{}]"
                    "[grouping:{}]"
                    "[strategy:{}]\n",
                    input, fastaFormat, markovianOrder,
                    k, criteria, grouping, strategy );

        for (auto &c : splitParameters( criteria ))
            for (auto &g : splitParameters( grouping ))
                for (auto &s: splitParameters( strategy ))
                    for (auto &o : splitParameters( markovianOrder ))
                    {
                        fmt::print( "[Params]"
                                    "[order:{}]"
                                    "[criteria:{}]"
                                    "[grouping:{}]"
                                    "[strategy:{}]\n", o, c, g, s );
                        std::visit( [&]( auto &&p ) {
                            p.runPipeline_VALIDATION( LabeledEntry::loadEntries( input, fastaFormat ),
                                                      std::stoi( o ), k );
                        }, getConfiguredPipeline( g, c, s ));
                    }


    }
    return 0;
}
