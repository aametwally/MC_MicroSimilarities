//
// Created by asem on 14/09/18.
//

#include "AbstractMC.hpp"

namespace MC
{
    template class AbstractMC< AAGrouping_OFER15  >;
    template class AbstractMC< AAGrouping_NOGROUPING22  >;

    template class ModelGenerator< AAGrouping_OFER15 >;
    template class ModelGenerator< AAGrouping_NOGROUPING22 >;
}