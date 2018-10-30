//
// Created by asem on 03/08/18.
//

#include "ConfusionMatrix.hpp"
#include "clara.hpp"

int main(int argc, char *argv[])
{
    const std::vector< int64_t > labels{2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2};
    const std::vector< int64_t > prediction{0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2};

    ConfusionMatrix c( std::set< int64_t >( labels.begin() , labels.cend()) );
    for( auto i = 0 ; i < std::min( labels.size() , prediction.size()) ; ++i )
    {
        c.countInstance(prediction[i] , labels[i]);
    }
    c.printReport<3>();
    c.printClassReport<5>(0);
    c.printClassReport<5>(1);
    c.printClassReport<5>(2);

    return 0;
}
