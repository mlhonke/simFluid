#include "sim_water.hpp"
#include "sim_label.hpp"
#include "sim_pls_cuda.hpp"

int main(int argc, char** argv){
    // Build the parameter structures to setup the simulation.
    SimParams* params; // Grid and global physics (gravity) parameters
    SimWaterParams* water_params; // Fluid parameters (viscosity, surface tension...)
    int n_steps;
    SimWater::create_params_from_args(argc, argv, n_steps, params, water_params);
    SimWater sim = SimWater(*params, *water_params);
//    sim.write_mesh = true;

    // Create scenario.
    CubeXi container(params->grid_w+2, params->grid_w+2, params->grid_w+2, arma::fill::zeros);
    container.subcube(0, 0, params->grid_w/2.0, params->grid_w+1, params->grid_h/2.0, params->grid_d+1).fill(SOLID);
    container.subcube(2, 2, params->grid_w/2.0+2, params->grid_w-1, params->grid_h/2.0-2, params->grid_d+1).fill(AIR);
    container.subcube(params->grid_w/2.0-params->grid_w/8.0+2, params->grid_h/2.0-2, params->grid_w/2.0+3,
            params->grid_w/2.0+params->grid_w/8.0, params->grid_h/2.0, params->grid_w/2.0+2+params->grid_d/8.0).fill(AIR);
    container.subcube(0, params->grid_w/2.0-1, 0, params->grid_w+1, params->grid_h+1, params->grid_d/2.0).fill(SOLID);
    container.subcube(2, params->grid_w/2.0+1, 2, params->grid_w-1, params->grid_h-1, params->grid_d/2.0).fill(AIR);
    sim.fluid_label->label.subcube(2, 2, 2, params->grid_w-1, params->grid_h-1, params->grid_d-1) =
            container.subcube(2, 2, 2, params->grid_w-1, params->grid_h-1, params->grid_d-1);

    // Output model for use with rendering.
    SimLabel::triangulate_grid(container, {-1,-1,-1}, SOLID, "../solid.ply", *params);

    // Choose a surface tracking method and make an initial surface.
    auto level_set = new SimPLSCUDA(*params, sim.DEV_C, sim.DEV_V);
//    level_set->reseed_interval = 1;
    Vector3 corner = {1, 1, params->grid_d/2.0+1};
    std::cout << corner << std::endl;
    Vector3 corner_2 = {params->grid_w-2.0, params->grid_h/2.0-3, params->grid_d-2.0};
    level_set->initialize_level_set_rectangle(corner, corner_2);
    sim.initialize_fluid(level_set);

    // Run the simulation.
    for (int i = 0; i < n_steps; i++){
        sim.step();
    }

    return 0;
}
