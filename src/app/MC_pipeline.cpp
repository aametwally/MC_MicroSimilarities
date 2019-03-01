
#include "Pipeline.hpp"
#include "clara.hpp"


std::vector<std::string> splitParameters( std::string params )
{
    io::trim( params, "[({})]" );
    return io::split( params, "," );
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
    std::string order = "3";

    size_t k = 10;
    bool showHelp = false;
    std::string grouping = GroupingLabel_NOGROUPING;
    std::string criteria = keys( CriteriaLabels ).front();
    std::string strategy = keys( ClassifierEnum ).front();
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
              | clara::Opt( criteria, join( keys( CriteriaLabels ), "|" ))
              ["-c"]["--criteria"]
                      ( fmt::format( "Similarity Criteria, default:{}", criteria ))
              | clara::Opt( strategy, join( keys( ClassifierEnum ), "|" ))
              ["-s"]["--strategy"]
                      ( fmt::format( "Classification Strategy, default:{}", strategy ))
              | clara::Opt( order, "MC order" )
              ["-o"]["--order"]
                      ( fmt::format( "Specify MC of higher order o, default:{}", order ))
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
        fmt::print( "[Args][input:{}][testFile:{}][model:{}][order:{}][kfold:{}]\n",
                    input, testFile, model, order, k );
    } else
    {
        fmt::print( "[Args][input:{}]"
                    "[fformat:{}]"
                    "[model:{}]"
                    "[order:{}]"
                    "[k-fold:{}]"
                    "[criteria:{}]"
                    "[grouping:{}]"
                    "[strategy:{}]\n",
                    input, fastaFormat, model, order,
                    k, criteria, grouping, strategy );

        for (auto &c : splitParameters( criteria ))
        {
            for (auto &m : splitParameters( model ))
            {
                for (auto &o : splitParameters( order ))
                {

                    for (auto &g : splitParameters( grouping ))
                    {

                        fmt::print( "[Params]"
                                    "[model:{}]"
                                    "[order:{}]"
                                    "[criteria:{}]"
                                    "[grouping:{}]\n", m, o, c, g );

                        std::visit( [&]( auto &&p ) {
                            p.runPipeline_VALIDATION( LabeledEntry::loadEntries( input, fastaFormat ), k,
                                                      splitParameters( strategy ));
                        }, getConfiguredPipeline( g, c, m, std::stoi( o )));
                    }
                }
            }
        }
    }
    return 0;
}
