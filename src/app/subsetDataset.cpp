#include "common.hpp"
#include "clara.hpp"

int main(int argc, char *argv[])
{
    std::string clustersFile;
    float percentage = 0.1f;
    float threshold = -1;

    auto cli
        = clara::Arg( clustersFile, "input" )
            ("UniRef input file")
        | clara::Opt( percentage, "percentage" )
            ["-p"]["--percentage"]
            ("subset percentage")
        | clara::Opt( threshold , "threshold" )
            ["-t"]["--t"]
            ("Below cluster size average to exclude from subsetting" );

    auto result = cli.parse( clara::Args( argc, argv ) );
    if( !result ) {
        fmt::print("Error in command line:{}\n", result.errorMessage() );
        exit(1);
    }

    fmt::print("[Args][input:{}][percentage:{}][threshold:{}]\n",
                clustersFile , percentage , threshold );

    auto seqs = io::readFastaFile( argv[1] );

    auto unirefItems = io::fasta2UnirefItems( seqs );
    seqs.clear();

    auto reducedAlphabetClusters = preprocess::unirefClusters2ReducedAASequences_DIAMOND( unirefItems );
    unirefItems.clear();

    auto clustersN = reducedAlphabetClusters.size();
    size_t populationN = 0;
    for( const auto &c : reducedAlphabetClusters )
        for( const auto &s : c )
            populationN += s.length();

    auto averageClusterSize = populationN / clustersN;

    fmt::print("[Clusters Size:{}][Average Cluster Size:{}][All Sequences:{}]\n", clustersN , averageClusterSize , populationN );

    auto results = markovianTraining( reducedAlphabetClusters , markovianOrder ,
                                      gappedMarkovianOrder , averageClusterSize , 0.1 );

    fmt::print("[Test Sequences:{}]\n",results.second.size());

    reducedAlphabetClusters.clear();
    seqs.clear();
    unirefItems.clear();

    auto classificationResults = classification::classify( results.second , results.first );
//    writeResults( results , argv[2] );


    return 0;
}
