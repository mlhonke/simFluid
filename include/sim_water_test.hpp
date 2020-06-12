//
// Created by graphics on 17/11/19.
//

#ifndef FERRO3D_SIM_WATER_TEST_HPP
#define FERRO3D_SIM_WATER_TEST_HPP

#include "sim_water.hpp"

class SimLabel;

class SimWaterTest : public SimWater{
public:
    SimWaterTest(SimParams &C, SimWaterParams &CW);

    void step();

private:
    bool testIsolatedFluidPressureSolve();
};


#endif //FERRO3D_SIM_WATER_TEST_HPP
