#include "markovian_features.hpp"
#include "clara.hpp"


int main( int argc, char *argv[] )
{
    std::string input, testFile;
    std::string fastaFormat = "uniref";
    int markovianOrder = 2;
    size_t k = 10;
    bool showHelp = false;
    std::string grouping = "diamond11";
    std::string criteria = "chi";
    std::string strategy = "totaldist";

    auto cli
            = clara::Arg( input, "input" )
                      ( "UniRef input file" )
              | clara::Opt( testFile, "test" )
              ["-T"]["--test"]
                      ( "test file" )
              | clara::Opt( grouping, "grouping" )
              ["-G"]["--grouping"]
                      ( "grouping method" )
              | clara::Opt( fastaFormat, "fastaformat" )
              ["-f"]["--fformat"]
              | clara::Opt( criteria, "criteria" )
              ["-c"]["--criteria"]
                      ( "Similarity Criteria" )
              | clara::Opt( strategy, "strategy" )
              ["-s"]["--strategy"]
                      ( "Classification Strategy" )
              | clara::Opt( markovianOrder, "order" )
              ["-o"]["--order"]
                      ( "Markovian order" )
              | clara::Opt( k, "k-fold" )
              ["-k"]["--k-fold"]
                      ( "cross validation k-fold" )
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
                    "[grouping:{}]\n",
                    input, fastaFormat, markovianOrder,
                    k,
                    criteria, grouping );

        std::visit( [&]( auto &&p ) {
            p.runPipeline_VALIDATION( UniRefEntry::loadEntries( input, fastaFormat ),
                                      markovianOrder,
                                      k );
        }, getConfiguredPipeline( grouping, criteria , strategy ));
    }
    return 0;
}
