#include "sim_water.hpp"
//#include "sim_levelset.hpp"
#include "sim_pls_cuda.hpp"

int main(int argc, char** argv){
    // Build the parameter structures to setup the simulation.
    SimParams* params; // Grid and global physics (gravity) parameters
    SimWaterParams* water_params; // Fluid parameters (viscosity, surface tension...)
    int n_steps;
    SimWater::create_params_from_args(argc, argv, n_steps, params, water_params);
    SimWater sim = SimWater(*params, *water_params);
    sim.write_mesh = true;

    // Choose a surface tracking method and make an initial surface.
    auto level_set = new SimPLSCUDA(*params, sim.DEV_C, sim.DEV_V);
    Vector3 corner = {params->grid_w/4.0-0.5, params->grid_h/4.0-0.5, params->grid_d/4.0-0.5};
    std::cout << corner << std::endl;
    Vector3 corner_offset = {params->grid_w/2.0, params->grid_h/2.0, params->grid_d/2.0};
    level_set->initialize_level_set_rectangle(corner, corner + corner_offset);
//    std::cout << level_set->LS << std::endl;
    sim.initialize_fluid(level_set);

    // Run the simulation.
    for (int i = 0; i < n_steps; i++){
        sim.step();
    }

    return 0;
}
