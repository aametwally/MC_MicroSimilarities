#include "markovian_features.hpp"
#include "clara.hpp"



int main( int argc, char *argv[] )
{
    std::string input, testFile;
    std::string fastaFormat = "uniref";
    int markovianOrder = 2;
    int gappedMarkovianOrder = -1;
    float testPercentage = 0.1f, threshold = -1;
    bool showHelp = false;
    std::string grouping = "diamond11";
    std::string criteria = "chi";

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
              | clara::Opt( markovianOrder, "order" )
              ["-o"]["--order"]
                      ( "Markovian order" )
              | clara::Opt( gappedMarkovianOrder, "gorder" )
              ["-g"]["--gorder"]
                      ( "Gapped markovian order" )
              | clara::Opt( testPercentage, "percentage" )
              ["-p"]["--percentage"]
                      ( "test percentage" )
              | clara::Opt( threshold, "threshold" )
              ["-t"]["--t"]
                      ( "Below cluster size average to exclude from subsetting" )
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
        fmt::print( "[Args][input:{}][testFile:{}][order:{}][testPercentage:{}][threshold:{}]\n",
                    input, testFile, markovianOrder, testPercentage, threshold );


    } else
    {
        fmt::print( "[Args][input:{}]"
                    "[fformat:{}]"
                    "[order:{}]"
                    "[testPercentage:{}]"
                    "[threshold:{}]"
                    "[criteria:{}]"
                    "[grouping:{}]\n",
                    input, fastaFormat , markovianOrder,
                    testPercentage, threshold,
                    criteria, grouping );

        std::visit( [&]( auto &&p ) {
            p.runPipeline_VALIDATION( UniRefEntry::loadEntries( input ,  fastaFormat ),
                                      markovianOrder,
                                      testPercentage,
                                      threshold );
        }, getConfiguredPipeline( grouping, criteria ));
    }
    return 0;
}
