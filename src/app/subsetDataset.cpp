#include "common.hpp"
#include "clara.hpp"

int main(int argc, char *argv[]) {
    std::string clustersFile, outputDir;
    float percentage = 0.1f;
    float threshold = -1;

    auto cli
            = clara::Arg(clustersFile, "input")
                      ("UniRef input file")
              | clara::Arg(outputDir, "output")
                      ("output directory")
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
               clustersFile, outputDir , percentage, threshold);

    auto seqs = io::readFastaFile(argv[1]);
    auto [test,training] = ( threshold > 0 )?
                           uniref::separationExcludingClustersWithLowSequentialData( seqs , percentage , threshold ) :
                           subsetRandomSeparation( seqs , percentage );

    return 0;
}
