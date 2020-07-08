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
    sim.write_mesh = true;

    // Create stairs
    sim.fluid_label->label.subcube(2, 2, 2, params->grid_h/4, params->grid_h-1, params->grid_d/2).fill(SOLID);
    for (int i = 0; i < params->grid_w/2-1; i++){
        sim.fluid_label->label.subcube(params->grid_w/4+i, 2, 2, params->grid_w/4+i+1, params->grid_h-1, params->grid_d/2-i).fill(SOLID);
    }
    std::cout << sim.fluid_label->label << std::endl;
    CubeXi stairs_portion = sim.fluid_label->label.subcube(2,2,2, params->grid_w-1, params->grid_h-1, params->grid_d-1);
    SimLabel::triangulate_grid(stairs_portion, {1,1,1}, SOLID, "../stairs.ply", *params);

    // Choose a surface tracking method and make an initial surface.
    auto level_set = new SimPLSCUDA(*params, sim.DEV_C, sim.DEV_V);
    Vector3 corner = {2, 2, params->grid_d/2.0};
    std::cout << corner << std::endl;
    Vector3 corner_2 = {params->grid_w/4.0, params->grid_h-1.0, params->grid_h-2.0};
    level_set->initialize_level_set_rectangle(corner, corner_2);
//    std::cout << level_set->LS << std::endl;
    sim.initialize_fluid(level_set);

    // Run the simulation.
    for (int i = 0; i < n_steps; i++){
        sim.step();
    }

    return 0;
}
