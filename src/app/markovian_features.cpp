#include "markovian_features.hpp"
#include "clara.hpp"
#include "Timers.hpp"

constexpr const char *LOADING = "loading";
constexpr const char *PREPROCESSING = "preprocessing";
constexpr const char *TRAINING = "training";
constexpr const char *CLASSIFICATION = "classification";

int main(int argc, char *argv[])
{
    std::string clustersFile;
    int markovianOrder = 2;
    int gappedMarkovianOrder = -1;
    float testPercentage = 0.1f, threshold = -1;
    bool showHelp = false;
    auto cli
            = clara::Arg( clustersFile, "input" )
            ("UniRef input file")
            | clara::Opt( markovianOrder, "order" )
            ["-o"]["--order"]
            ("Markovian order")
            | clara::Opt( gappedMarkovianOrder , "gorder" )
            ["-g"]["--gorder"]
            ("Gapped markovian order" )
            | clara::Opt( testPercentage, "percentage" )
            ["-p"]["--percentage"]
            ("test percentage")
            | clara::Opt( threshold , "threshold" )
            ["-t"]["--t"]
            ("Below cluster size average to exclude from subsetting" )
            | clara::Help(showHelp);

    auto result = cli.parse( clara::Args( argc, argv ) );
    if( !result ) {
        fmt::print("Error in command line:{}\n", result.errorMessage() );
        exit(1);
    }
    else if (showHelp)
    {
        cli.writeToStream(std::cout);
    }
    else
    {
        fmt::print("[Args][input:{}][order:{}][testPercentage:{}][threshold:{}]\n",
                   clustersFile , markovianOrder, testPercentage ,  threshold );

        Timers::tic( LOADING );
        std::vector< UniRefEntry > entries = UniRefEntry::fasta2UnirefEntries( FastaEntry::readFastaFile( argv[1] ) );
        Timers::toc( LOADING );
        Timers::report_s( LOADING );


        auto [trainingClusters , test] = Timers::reported_invoke_s( [&](){

            entries = preprocess::reducedAAEntries_DIAMOND( entries );

            fmt::print("[All Sequences:{}]\n", entries.size());

            auto [test,training] = ( threshold > 0 )?
                                   UniRefEntry::separationExcludingClustersWithLowSequentialData( entries , testPercentage , threshold ) :
                                   subsetRandomSeparation( entries , testPercentage );

            fmt::print("[Training Entries:{}][Test Entries:{}][Test Ratio:{}]\n",
                       training.size(),test.size(), float{test.size()}/entries.size());

            entries.clear();

            auto trainingClusters = UniRefEntry::groupSequencesByUniRefClusters( training );
            return std::make_pair( trainingClusters , test );
        }, PREPROCESSING );


        fmt::print("[Training Clusters:{}]\n" ,  trainingClusters.size());
        auto trainedProfiles = Timers::reported_invoke_s( [&](){
            return markovianTraining( trainingClusters , markovianOrder );
        } , TRAINING );

        auto classificationResults = Timers::reported_invoke_s( [&](){
            return classification::classify_VALIDATION( test , trainedProfiles );
        }, CLASSIFICATION );
    }
    return 0;
}
