#ifndef SIM_PARAMS_WATER_HPP
#define SIM_PARAMS_WATER_HPP

#include "sim_params.hpp"

//Enumerated types here
typedef enum {
    AIR = 0,
    FLUID = 1,
    SOLID = 2,
    FLUID_CENTER = 3
} FLUID_TYPES;

typedef struct SimWaterParams {
    SimWaterParams(){}

    SimWaterParams(scalar_t density, scalar_t lambda, scalar_t nu, scalar_t g) :
        density(density), lambda(lambda), nu(nu), g(g)
    {}

    scalar_t const density = 1000.0;
    scalar_t const lambda = 0.0;
    scalar_t const nu = 0.0;
    scalar_t const g = -9.8;
} SimWaterParams;

#endif // SIM_PARAMS_WATER_HPP
