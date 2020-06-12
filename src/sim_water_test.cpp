//
// Created by graphics on 17/11/19.
//

#include "sim_water_test.hpp"
#include "sim_label.hpp"
#include "sim_pls_cuda.cuh"

SimWaterTest::SimWaterTest(SimParams &C, SimWaterParams &CW)
    : SimWater(C, CW)
{
    // Run single step tests
    testIsolatedFluidPressureSolve();
}

void SimWaterTest::step(){

}

bool SimWaterTest::testIsolatedFluidPressureSolve() {
    SimLabel testLabels(*this, true);

    testLabels.label.fill(2);
    testLabels.label.subcube(2, 2, 2, grid_w-1, grid_h-1, grid_d-1).fill(0);
    testLabels.label.subcube(2, 2, 2, grid_w-1, grid_h-1, grid_d/2).fill(1);
    testLabels.label.subcube(2, 2, 2, 8, 8, 8).fill(2);
//    testLabels.label(3, 3, 3) = 1;
//    testLabels.label(4, 4, 4) = 1;
//    testLabels.label(4, 5, 4) = 1;
//    testLabels.label(4, 4, 5) = 1;
//    testLabels.label(4, 4, 6) = 1;
//    testLabels.label(4, 4, 7) = 1;
    testLabels.label.subcube(4, 4, 4, 5, 5, 5).fill(1);

    // Need a corresponding levelset to use with labels. Set up interfaces then force a redistancing.
    for (int k = 0; k < grid_d; k++){
        for (int j = 0; j < grid_h; j++){
            for (int i = 0; i < grid_w; i++){
                if (testLabels.get_label_center(i,j,k) == 1){
                    simLS->LS(i,j,k) = -1;
                } else {
                    simLS->LS(i,j,k) = 1;
                }
            }
        }
    }

    simLS->redistance_interface();
    simLS->redistance();
    std::cout << "Level set" << std::endl;
    std::cout << simLS->LS << std::endl;

    testLabels.colour_label(simLS->LS);
    std::cout << "Colours" << std::endl;
    std::cout << testLabels.colours << std::endl;

    V[2].fill(-1.0);

    solve_pressure(&testLabels, false, false, true);

    for (const auto& has_air : testLabels.has_air){
        std::cout << has_air << std::endl;
    }
    for (const auto& center : testLabels.centers){
        std::cout << center << std::endl;
    }

    return true;
}