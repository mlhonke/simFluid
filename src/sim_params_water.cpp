//
// Created by graphics on 2020-06-22.
//

#include "sim_params_water.hpp"
#include <iostream>

void create_water_params_from_args(int argc, char **argv, SimWaterParams *&retCW, int &i){
    if (argc > 1) {
        retCW = new SimWaterParams{
                std::stof(argv[i++]),
                std::stof(argv[i++]),
                std::stof(argv[i++]),
                std::stof(argv[i++])
        };
    } else {
        retCW = new SimWaterParams();
    }
}