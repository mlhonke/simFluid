#include "sim_water.hpp"
#include "sim_label.hpp"
//#include "sim_levelset.hpp"
#include "sim_pls_cuda.hpp"

int main(int argc, char** argv){
    // Build the parameter structures to setup the simulation.
    SimParams* params; // Grid and global physics (gravity) parameters
    SimWaterParams* water_params; // Fluid parameters (viscosity, surface tension...)
    int n_steps;
    SimWater::create_params_from_args(argc, argv, n_steps, params, water_params);
    SimWater sim = SimWater(*params, *water_params);
    sim.write_mesh = false;

    // Create stairs.
    sim.fluid_label->label.subcube(2, 2, 2, params->grid_h/4, params->grid_h-1, params->grid_d/2+1).fill(SOLID);
    int n_stair_steps = 6;
    int stair_case_height = params->grid_d/2;
    int stair_case_width = params->grid_w/2;
    int step_height = stair_case_height / n_stair_steps;
    int step_width = stair_case_width / n_stair_steps;
    for (int i = 0; i < n_stair_steps; i++){
        sim.fluid_label->label.subcube(params->grid_w/4+i*step_width, 2, 1,
                params->grid_w/4+(i+1)*step_width, params->grid_h-1, params->grid_d/2-(i+1)*step_height+1).fill(SOLID);
    }

    // Output model of stairs for later use in rendering.
    CubeXi one_side_open = sim.fluid_label->label.subcube(0,2,0,params->grid_w+1, params->grid_h+1, params->grid_d-1);
    SimLabel::triangulate_grid(one_side_open, {-1,1,-1}, SOLID, "../solid.ply", *params);

    // Choose a surface tracking method and make an initial surface.
    auto level_set = new SimPLSCUDA(*params, sim.DEV_C, sim.DEV_V);
//    level_set->reseed_interval = 1;
    Vector3 corner = {1, 1, params->grid_d/2.0+1};
    std::cout << corner << std::endl;
    Vector3 corner_2 = {params->grid_w/4.0, params->grid_h-1.0, params->grid_h-2.0};
    level_set->initialize_level_set_rectangle(corner, corner_2);
    sim.initialize_fluid(level_set);

    // Run the simulation.
    for (int i = 0; i < n_steps; i++){
        sim.step();
    }

    return 0;
}
