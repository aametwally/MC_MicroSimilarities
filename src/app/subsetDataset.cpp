#include "FastaEntry.hpp"
#include "common.hpp"
#include "clara.hpp"

int main(int argc, char *argv[]) {
    std::string clustersFile, outputPrefix;
    float percentage = 0.1f;
    float threshold = -1;

    auto cli
            = clara::Arg(clustersFile, "input")
                      ("UniRef input file")
              | clara::Arg(outputPrefix, "outputPrefix")
                      ("output prefix")
              | clara::Opt(percentage, "percentage")
              ["-p"]["--percentage"]
                      ("subset percentage")
              | clara::Opt(threshold, "threshold")
              ["-t"]["--t"]
                      ("Below cluster size average to exclude from subsetting");

    auto result = cli.parse(clara::Args(argc, argv));
    if (!result) {
        fmt::print("Error in command line:{}\n", result.errorMessage());
        exit(1);
    }

    fmt::print("[Args][input:{}][output:{}][percentage:{}][threshold:{}]\n",
               clustersFile, outputPrefix , percentage, threshold);

    auto seqs = FastaEntry::readFastaFile( clustersFile );
    auto [test,training] = ( threshold > 0 )?
                           FastaEntry::separationExcludingClustersWithLowSequentialData( seqs , percentage , threshold ) :
                           subsetRandomSeparation( seqs , percentage );

    FastaEntry::writeFastaFile( test , fmt::format("{}_test.fasta",outputPrefix));
    FastaEntry::writeFastaFile( training , fmt::format("{}_training.fasta",outputPrefix));

    return 0;
}
