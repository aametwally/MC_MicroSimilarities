#include "markovian_features.hpp"

int main(int argc, char *argv[])
{
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

    std::cout << "Average Cluster Size:" << averageClusterSize << std::endl;

    auto results = markovianTraining( reducedAlphabetClusters , 3 , 2 , averageClusterSize , 0.1 );
    reducedAlphabetClusters.clear();
    seqs.clear();
    unirefItems.clear();

    writeResults( results , argv[2] );


    return 0;
}
