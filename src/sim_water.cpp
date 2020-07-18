#include <iostream>
#include <cmath>
#include <algorithm>

#include "interpolate.hpp"
#include "sim_water.hpp"
#include "fd_math.hpp"
#include "sim.hpp"
#include "advect.hpp"
#include "sim_levelset.hpp"
#include "sim_label.hpp"
#include "marchingtets.h"
#include "CudaCG.hpp"
#include "sim_viewer.hpp"

//static Eigen::ConjugateGradient<MatrixA, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double>> CG;
static Eigen::ConjugateGradient<MatrixA, Eigen::Lower|Eigen::Upper> CG;

SimWater::SimWater(SimParams C, SimWaterParams CW):Sim(C), density(CW.density), lambda(CW.lambda), nu(CW.nu), g(CW.g){
    // Want to use Eigen for Poisson solve, but Armadillo for everything else since need 3D representation.
    p = VectorXs::Zero(n_cells, 1);
    P = CubeX(p.data(), grid_w, grid_h, grid_d, false);
    air_label = new SimLabel(*this, true);

    // Setup the simulation domain ( solid boundaries, air in center )
    fluid_label->label.fill(SOLID);
    fluid_label->label.subcube(2, 2, 2, grid_w-1, grid_h-1, grid_d-1).fill(AIR);

    // Initialize pressure solve data.
    A = MatrixA(n_cells, n_cells);
    Ad = MatrixA(n_cells, n_cells);
    b = VectorXs::Zero(n_cells, 1);

    // Initialize viscosity solve data.
    A_vis_u = MatrixA((grid_w+1)*grid_h*grid_d, (grid_w+1)*grid_h*grid_d);
    A_vis_v = MatrixA(grid_w*(grid_h+1)*grid_d, grid_w*(grid_h+1)*grid_d);
    A_vis_w = MatrixA(grid_w*grid_h*(grid_d+1), grid_w*grid_h*(grid_d+1));
    A_vis_ud = MatrixA((grid_w+1)*grid_h*grid_d, (grid_w+1)*grid_h*grid_d);
    A_vis_vd = MatrixA(grid_w*(grid_h+1)*grid_d, grid_w*(grid_h+1)*grid_d);
    A_vis_wd = MatrixA(grid_w*grid_h*(grid_d+1), grid_w*grid_h*(grid_d+1));
    u_new = VectorXs::Zero((grid_w+1)*grid_h*grid_d);
    v_new = VectorXs::Zero(grid_w*(grid_h+1)*grid_d);
    w_new = VectorXs::Zero(grid_w*grid_h*(grid_d+1));

    sim_viewer = new SimViewer(C);
    sim_viewer->launch();
//    load_data(); // Restore a previous simulation if available
}

void SimWater::initialize_fluid(SimLevelSet *level_set){
    simLS = level_set;
    tets = new marchingtets::MarchingTets(&(simLS->LS), {0,0,0}, scale_w);
    update_labels_from_level_set();
    update_triangle_mesh();
    update_viewer_triangle_mesh();
    volume_old = -calc_mesh_volume(tets->x, tets->tri);
    int n_cells_occupied = fluid_label->get_cell_count(1); // Find number of fluid cells currently.
    n_cells_use = 7*n_cells_occupied; //Upper bound.
    cudacg_water = new CudaCG(n_cells_occupied, n_cells_use);
//    cudacg_water->project = true;
    cudacg_vis = new CudaCG(n_cells_occupied, n_cells_use);
}

void SimWater::create_params_from_args(int argc, char **argv, int &n_steps, SimParams *&retC, SimWaterParams *&retCW) {
    int i;
    create_params_from_args(argc, argv, n_steps, retC, retCW, i);
}

void SimWater::create_params_from_args(int argc, char **argv, int &n_steps, SimParams *&retC, SimWaterParams *&retCW, int &i) {
    // Create param structures on heap since they live the life of the simulation.
    i = 1;
    if (argc > 1) {
        n_steps = std::stoi(argv[i++]);
    }
    else {
        n_steps = 1000000;
    }
    create_sim_params_from_args(argc, argv, retC, i);
    create_water_params_from_args(argc, argv, retCW, i);
}

void SimWater::save_data(){
    Sim::save_data();
    simLS->save_data();
}

void SimWater::load_data(){
    Sim::load_data();
    simLS->load_data();
}

void SimWater::step(){
    ExecTimerSteps timer("Simulation time", false);
    if (cur_step % 10 == 0){
//        save_data();
    }

    elapsed_time += dt;
    dt = get_new_timestep(V);
    if (elapsed_time + dt >= render_time - 1E-14) {
        dt = render_time - elapsed_time;
        render_time += render_dt;
        do_render = true;
    }
    std::cout << "Time Step = " << dt << " Time I" << cur_step << " Total elapsed: " << elapsed_time << " Render Step: " << do_render << std::endl;
    timer.next("Completed time step size calculation");

    extrapolate_velocities_from_LS();
    timer.next("Extrapolate velocities");
    update_velocities_on_device();
    timer.next("Send velocities to device");

    simLS->advance(cur_step, dt);
    timer.next("Advance level set");

    update_labels_from_level_set();
    timer.next("Updating simulation labels");
    advect_velocity();
    timer.next("Advecting velocity");
    add_gravity_to_velocity(V[2], dt);
    timer.next("Add gravity to velocity");
    std::cout << "Solving viscosity." << std::endl;
    solve_viscosity();
    timer.next("Solving viscosity");
    std::cout << "Solving pressure." << std::endl;
    solve_pressure(fluid_label, true, false, false);
    timer.next("Solving pressure");

    cur_step++;
    if (do_render) {
        update_triangle_mesh();
        update_viewer_triangle_mesh();
        do_render = false;
        timer.next("Rendering screen");
    }
}

void SimWater::add_gravity_to_velocity(CubeX &v, scalar_t dt){
    for (unsigned int k = 0; k < v.n_slices; k++) {
        for (unsigned int j = 0; j < v.n_cols; j++) {
            for (unsigned int i = 0; i < v.n_rows; i++) {
               // if (fluid_label->get_label_center(i, j, k) == 1 || fluid_label->get_label_center(i, j, k-1) == 1) {
                    v(i, j, k) += g * dt;
               // }
            }
        }
    }
}

void SimWater::extrapolate_velocities_from_LS() {
    CubeX divs = CubeX(grid_w, grid_h, grid_d, arma::fill::zeros);

    extrapolate_velocity_from_LS(V[0], {-1, 0, 0});
    extrapolate_velocity_from_LS(V[1], {0, -1, 0});
    extrapolate_velocity_from_LS(V[2], {0, 0, -1});

    set_boundary_velocities();

    bool do_pressure_extrap = true;
    if (do_pressure_extrap) {
        V_solid[0] = V[0];
        V_solid[1] = V[1];
        V_solid[2] = V[2];
        update_labels_for_air();

        air_label->colour_label(simLS->LS);

        solve_pressure(air_label, false, false, true);

        V_solid[0].fill(0);
        V_solid[1].fill(0);
        V_solid[2].fill(0);
    }

//    check_divergence(divs);
}

void SimWater::advect_velocity(){
    macVel V_new = V;
    advect_RK3_CUDA(V_new[0], {0.5, 0.0, 0.0}, DEV_V, dt, DEV_C, true, true);
    advect_RK3_CUDA(V_new[1], {0.0, 0.5, 0.0}, DEV_V, dt, DEV_C, true, true);
    advect_RK3_CUDA(V_new[2], {0.0, 0.0, 0.5}, DEV_V, dt, DEV_C, true, true);
    V = V_new;
}

void SimWater::update_viewer_triangle_mesh(){
    sim_viewer->wait_for_viewer();

    sim_viewer->Fmesh.resize(tets->tri.size(), 3);
    sim_viewer->Vmesh.resize(tets->x.size(), 3);
    sim_viewer->Nmesh.resize(tets->x.size(), 3);

    int k = 0;
    for (auto x : tets->x){
        Vector3 n = get_grad_lerped(x, simLS->LS, scale_w);
        sim_viewer->Vmesh(k, 0) = x[0];
        sim_viewer->Vmesh(k, 1) = x[1];
        sim_viewer->Vmesh(k, 2) = x[2];
        sim_viewer->Nmesh(k, 0) = n[0];
        sim_viewer->Nmesh(k, 1) = n[1];
        sim_viewer->Nmesh(k, 2) = n[2];
        k++;
    }

    k = 0;
    for (auto t : tets->tri) {
        sim_viewer->Fmesh(k, 0) = t[2];
        sim_viewer->Fmesh(k, 1) = t[1];
        sim_viewer->Fmesh(k, 2) = t[0];
        k++;
    }

    if (write_mesh) {
        sim_viewer->write_mesh_to_ply();
    }
    sim_viewer->refresh_viewer();
}

void SimWater::update_triangle_mesh(){
    tets->tri.clear();
    tets->x.clear();
    tets->edge_cross.clear();
    tets->cube_record.clear();

    // Loop through all the cubes, this is one less in each dimension then there is grid points.
    for (int k = 0; k < grid_d-1; k++){
        for (int j = 0; j < grid_h-1; j++){
            for (int i = 0; i < grid_w-1; i++){
                tets->contour_cube(i,j,k);
            }
        }
    }
}

void SimWater::solve_pressure(const SimLabel *labels_in, bool do_tension, bool do_vol_correct, bool check_air) {
    scalar_t volume_delta = 0;
    scalar_t volume = 0;
    scalar_t c = 0;

    if (do_vol_correct) {
        update_triangle_mesh();
        volume = -calc_mesh_volume(tets->x, tets->tri);
        volume_delta = volume - volume_old;
        std::cout << "Volume difference " << volume_delta << std::endl;
        std::cout << "New volume is " << volume << std::endl;
        scalar_t x = volume_delta / volume_old;
        scalar_t kp = 10.0 / (25.0*dt);
        c = kp*(x/(x+1.0));
    }

    A.setZero();
    Ad.setZero();
    b.fill(0);
    p.fill(0);
    A.reserve(Eigen::VectorXi::Constant(grid_w*grid_h*grid_d, 7)); // 6 faces, and center
    Ad.reserve(Eigen::VectorXi::Constant(grid_w*grid_h*grid_d, 1));

    build_A_and_b(A, b, dt, simLS->LS, c, labels_in, do_tension, check_air);
    A.makeCompressed();
    Ad.makeCompressed();
    int A_n_rows = A.rows();

#ifdef USECUDA
    // The number of non-zero entries in Ad is equal to the number of fluid containing cells, hence actual rows in A.
    cudacg_water->load_matrix(A.outerIndexPtr(), A.innerIndexPtr(), A.valuePtr(), p.data(), b.data(), A.rows(), A.nonZeros());
    cudacg_water->load_diagonal(Ad.outerIndexPtr(), Ad.innerIndexPtr(), Ad.valuePtr());
    cudacg_water->solve();
    cudacg_water->get_error();
    if(cudacg_water->k == 1){
        std::cout << "Solver failure, check log files." << std::endl;
        logfile << "Solver may have failed due to minimal iterations." << std::endl;
        logfile << "Labels" << "\n" << labels_in->label << std::endl;
        logfile << "Air labels coloured into regions." << std::endl;
        logfile << labels_in->colours << std::endl;
        for (const auto& has_air : labels_in->has_air){
            logfile << has_air << std::endl;
        }
        for (const auto& center : labels_in->centers){
            logfile << center << std::endl;
        }
    }
#else
    CG.setTolerance(1E-16);
    CG.compute(A);

    p = CG.solve(b);
    if(CG.info() != Eigen::Success){
        std::cout << "Pressure solve failed!" << std::endl;
        std::ofstream badsolvefile;
        badsolvefile.open("badsolvepressure.txt");
        badsolvefile << A << "\n";
        badsolvefile << b << "\n";
        badsolvefile.close();

//        Eigen::MatrixXd A_dense;
//        A_dense = Eigen::MatrixXd(A);
//        Eigen::EigenSolver<Eigen::MatrixXd> es;
//        es.compute(A_dense, false);
////        std::cout << "Eigenvalues are " << es.eigenvalues() << std::endl;
//        for (int i = 0; i < n_cells; i++){
//            scalar_t e_real = es.eigenvalues()[i].real();
//            if (e_real < 0){
//                std::cout << e_real << std::endl;
//            }
//        }

        std::cout << "velocites before solve were " << std::endl;
        std::cout << V[0] << std::endl;
        std::cout << V[1] << std::endl;
        std::cout << V[2] << std::endl;

        std::cout << "b was " << std::endl;
        std::cout << b << std::endl;
    }
    std::cout << "Stats for last pressure solve." << std::endl;
    std::cout << CG.error() << std::endl;
    std::cout << CG.iterations() << std::endl;
//    std::cout << "Pressure as solved for." << std::endl;
//    std::cout << P << std::endl;
#endif
    pressure_gradient_update(P, dt, simLS->LS, labels_in, do_tension, check_air);
}

void SimWater::solve_viscosity(){
    solve_viscosity_vel(A_vis_u, A_vis_ud, u_new, V[0], {-0.5, 0, 0});
    solve_viscosity_vel(A_vis_v, A_vis_vd, v_new, V[1], {0, -0.5, 0});
    solve_viscosity_vel(A_vis_w, A_vis_wd, w_new, V[2], {0, 0, -0.5});
}

// TODO: Make solve faster by not using it to just copy old velocity when in air.
void SimWater::solve_viscosity_vel(MatrixA &A_vis, MatrixA &A_visd, VectorXs &vel_new, CubeX &vel, const Vector3 &offset){
    A_vis.setZero();
    A_visd.setZero();
    A_vis.reserve(Eigen::VectorXi::Constant(vel.n_rows*vel.n_cols*vel.n_slices, 7));
    A_visd.reserve(Eigen::VectorXi::Constant(vel.n_rows*vel.n_cols*vel.n_slices, 1));
    VectorXs b_vis = Eigen::Map<VectorXs>(vel.memptr(), vel.n_elem); // Should be cheap because only map.
    build_A_and_b_viscosity(A_vis, b_vis, A_visd, vel.n_rows, vel.n_cols, vel.n_slices, offset);
//    std::cout << "A vis" << std::endl;
//    std::cout << A_vis << std::endl;
    A_vis.makeCompressed();
    A_visd.makeCompressed();

    cudacg_vis->load_matrix(A_vis.outerIndexPtr(), A_vis.innerIndexPtr(), A_vis.valuePtr(), vel_new.data(), b_vis.data(), A_vis.rows(), A_vis.nonZeros());
    cudacg_vis->load_diagonal(A_visd.outerIndexPtr(), A_visd.innerIndexPtr(), A_visd.valuePtr());
    cudacg_vis->solve();
    cudacg_vis->get_error();

//    CG.compute(A_vis);
//    vel_new = CG.solve(b_vis);
//    if(CG.info() != Eigen::Success){ std::cout << "Viscosity Solve Failed!" << std::endl;}
    vel = CubeX(vel_new.data(), vel.n_rows, vel.n_cols, vel.n_slices, false); // maybe don't copy once tested
}

void SimWater::extrapolate_velocity_from_LS(CubeX &v, Vector3i face){
    int max_int = 100000;
    CubeXi m(v.n_rows, v.n_cols, v.n_slices);
    m.fill(max_int);

    for (unsigned int k = 0; k < m.n_slices; k++) {
        for (unsigned int j = 0; j < m.n_cols; j++) {
            for (unsigned int i = 0; i < m.n_rows; i++) {
                if (fluid_label->get_label_center(i, j, k) == 1 || fluid_label->get_label_center(i + face(0), j + face(1), k + face(2)) == 1){
                    m(i,j,k) = 0;
                }
            }
        }
    }

    std::queue<Vector3ui> W;
    std::array<int, 6> neighbours {0, 0, 0, 0, 0, 0};
    for (unsigned int k = 0; k < m.n_slices; k++) {
        for (unsigned int j = 0; j < m.n_cols; j++) {
            for (unsigned int i = 0; i < m.n_rows; i++) {
                if (m(i,j,k) != 0){
                    int n_neighbours = get_neighbours_all<CubeXi, int>(neighbours, m, i, j, k);
                    for (int i_neighbour = 0; i_neighbour < n_neighbours; i_neighbour++){
                        if (neighbours[i_neighbour] == 0){
                            m(i,j,k) = 1;
                            W.push({i,j,k});
                            break;
                        }
                    }
                }
            }
        }
    }

    std::array<Vector3ui, 6> nc;
    while(!W.empty()){
        Vector3ui I = W.front();
        W.pop();
        int n_neighbours = get_neighbours_all<CubeXi, int>(neighbours, m, I(0), I(1), I(2));
        get_neighbours_coords<CubeXi>(nc, m, I(0), I(1), I(2));
        scalar_t sum = 0;
        unsigned int n_valid = 0;
        int d = m(I(0), I(1), I(2));

        for (int i = 0; i < n_neighbours; i++){
            if (neighbours[i] < d){
                sum += v(nc[i](0), nc[i](1), nc[i](2));
                n_valid += 1;
            } else if (neighbours[i] == max_int){
                W.push({nc[i](0), nc[i](1), nc[i](2)});
                m(nc[i](0), nc[i](1), nc[i](2)) = d + 1;
            }
        }

        v(I(0), I(1), I(2)) = sum / (scalar_t) n_valid;
    }
}

void SimWater::set_boundary_velocities(){
    for (unsigned int k = 0; k < grid_d; k++) {
        for (unsigned int j = 0; j < grid_h; j++) {
            for (unsigned int i = 0; i < grid_w; i++){
                if (fluid_label->get_label_center(i,j,k) == 2){
                    V[0](i,j,k) = 0;
                    V[0](i+1,j,k) = 0;
                    V[1](i,j,k) = 0;
                    V[1](i,j+1,k) = 0;
                    V[2](i,j,k) = 0;
                    V[2](i,j,k+1) = 0;
                }
            }
        }
    }
}

void SimWater::update_labels_from_level_set(){
    for (unsigned int k = 0; k < grid_d; k++) {
        for (unsigned int j = 0; j < grid_h; j++) {
            for (unsigned int i = 0; i < grid_w; i++){
                if (fluid_label->label(i+1, j+1, k+1) != 2){
                    if (simLS->LS(i,j,k) < 0){
                        fluid_label->label(i+1, j+1, k+1) = 1;
                    } else {
                        fluid_label->label(i+1, j+1, k+1) = 0;
                    }
                }
            }
        }
    }

    fluid_label->update_label_on_device();
}

void SimWater::update_labels_for_air(){
    air_label->label.subcube(2, 2, 2, grid_w-1, grid_h-1, grid_d-1).fill(0);
    std::array<int, 6> neighbours;
    std::array<Vector3ui, 6> n_coords;
    std::queue<Vector3ui> layer_coords;
    CubeXi m(grid_w+2, grid_h+2, grid_d+2, arma::fill::zeros);
    int n;
    for (unsigned int k = 0; k < grid_d; k++) {
        for (unsigned int j = 0; j < grid_h; j++) {
            for (unsigned int i = 0; i < grid_w; i++) {
                int il = i+1;
                int jl = j+1;
                int kl = k+1;
                if(fluid_label->label(il, jl, kl) == 1){
                    air_label->label(il,jl,kl) = 2;
                    n = get_neighbours_all<CubeXi, int>(neighbours, fluid_label->label, il, jl, kl);
                    get_neighbours_coords<CubeXi>(n_coords, fluid_label->label, il, jl, kl);
                    for (int ni = 0; ni < n; ni++){
                        if(neighbours[ni] == 0){
                            Vector3ui air_coord = n_coords[ni];
                            air_label->label(air_coord[0], air_coord[1], air_coord[2]) = 1;
                            m(air_coord[0], air_coord[1], air_coord[2])++;
                            layer_coords.push(air_coord);
                        }
                    }
                } else if (fluid_label->label(il, jl, kl) == SOLID){
                    air_label->label(il, jl, kl) = SOLID;
                }
            }
        }
    }

    while (!layer_coords.empty()){
        Vector3ui I = layer_coords.front();
        layer_coords.pop();
        int m_cur = m(I(0), I(1), I(2));
        n = get_neighbours_all<CubeXi, int>(neighbours, fluid_label->label, I(0), I(1), I(2));
        get_neighbours_coords<CubeXi>(n_coords, fluid_label->label, I(0), I(1), I(2));
        for (int i = 0; i < n; i++){
            Vector3ui air_coord = n_coords[i];
            if (neighbours[i] == 0 && m(air_coord[0], air_coord[1], air_coord[2]) == 0){
                air_label->label(air_coord[0], air_coord[1], air_coord[2]) = 1;
                if (m_cur < 2) {
                    m(air_coord[0], air_coord[1], air_coord[2]) = m_cur+1;
                    layer_coords.push(air_coord);
                }
            }
        }
    }
}

Vector3 SimWater::get_interface_position_between_cells(const Vector3ui &wet, const Vector3ui &dry, const CubeX &LS, scalar_t &theta){
    theta = make_non_zero(LS(wet(0), wet(1), wet(2))/(LS(wet(0), wet(1), wet(2)) - LS(dry(0), dry(1), dry(2))));
    scalar_t x = wet(0)*scale_w;
    scalar_t y = wet(1)*scale_h;
    scalar_t z = wet(2)*scale_d;
    if (wet(0) != dry(0))
        x =  (1-theta)*(wet(0)*scale_w) + theta*((dry(0))*scale_w);
    else if (wet(1) != dry(1))
        y =  (1-theta)*(wet(1)*scale_h) + theta*((dry(1))*scale_h);
    else if (wet(2) != dry(2))
        z =  (1-theta)*(wet(2)*scale_d) + theta*((dry(2))*scale_d);

    return {x, y, z};
}

/* build_A_element: Determine the coefficient for a row entry of A
 * ISC: The spatial index of the current (row's) cell
 * IS: The spatial index of the surrounding cell
 * IA: The index of the matrix
 * vel: The velocity for the respective face
 */
void SimWater::build_A_element(const Vector3ui &ISC, const Vector3ui &IS, const Vector2ui &IA, scalar_t vel,
        scalar_t vel_solid, MatrixA &A, VectorXs &b, scalar_t scale, const CubeX &LS, unsigned int b_index,
        const SimLabel *label_in, bool do_tension){
    if (label_in->get_label_center(IS(0), IS(1), IS(2)) == FLUID || label_in->get_label_center(IS(0), IS(1), IS(2)) == FLUID_CENTER){
        A.coeffRef(IA(0), IA(0)) += scale;
        A.coeffRef(IA(0), IA(1)) -= scale;
    } else if (label_in->get_label_center(IS(0), IS(1), IS(2)) == AIR){
        if (do_tension) {
            scalar_t theta;
            Vector3 interface = get_interface_position_between_cells(ISC, IS, LS, theta);
            scalar_t curvature = lambda * simLS->get_curvature(interface);
            A.coeffRef(IA(0), IA(0)) += scale * (1 - theta) / theta;
            b(b_index) += curvature / theta;
        }
        A.coeffRef(IA(0), IA(0)) += scale;
    } else if (label_in->get_label_center(IS(0), IS(1), IS(2)) == SOLID){
        b(b_index) += (1.0/scale_w)*(vel + vel_solid);
    }
}

//TODO: Reserve correct number of elements per row of sparse matrix.
void SimWater::build_A_and_b(MatrixA &A, VectorXs &b, scalar_t dt, const CubeX &LS, scalar_t c, const SimLabel *label_in, bool do_tension, bool check_air){
    scalar_t scale = dt / (scale_h*scale_w*density);
    const scalar_t MAX_P = 100;

    for (unsigned int i = 0; i < grid_w*grid_h*grid_d; i++){
        // Spatial coordinates
        unsigned int is = i % grid_w;
        unsigned int js = (i / grid_w) % grid_h;
        unsigned int ks = i / (grid_w*grid_h);
//        std::cout << is << " " << js << " " << ks << std::endl;
        unsigned int b_index = is + js*grid_w + ks*(grid_w*grid_h); // same as i here, redundant.
//        unsigned int b_index = i;
        if (label_in->get_label_center(is, js, ks) == FLUID){
            // RHS divergence of old velocity (c is for divergence control, mass conservation update).
            b(b_index) =     (-1.0/scale_w)*(V[0](is+1, js, ks) - V[0](is,js,ks))
                                  + (-1.0/scale_h)*(V[1](is, js+1, ks) - V[1](is,js,ks))
                                  + (-1.0/scale_d)*(V[2](is, js, ks+1) - V[2](is,js,ks)) - c;

            // 6 other cells / faces to reference for this row
            build_A_element({is, js, ks}, {is-1, js, ks}, {i, i-1},
                    -V[0](is, js, ks), V_solid[0](is, js, ks), A, b, scale, LS, b_index, label_in, do_tension);
            build_A_element({is, js, ks}, {is+1, js, ks}, {i, i+1},
                    V[0](is+1, js, ks), -V_solid[0](is+1, js, ks), A, b, scale, LS, b_index, label_in, do_tension);
            build_A_element({is, js, ks}, {is, js-1, ks}, {i, i-grid_w},
                    -V[1](is, js, ks), V_solid[1](is, js, ks), A, b, scale, LS, b_index, label_in, do_tension);
            build_A_element({is, js, ks}, {is, js+1, ks}, {i, i+grid_w},
                    V[1](is, js+1, ks), -V_solid[1](is, js+1, ks), A, b, scale, LS, b_index, label_in, do_tension);
            build_A_element({is, js, ks}, {is, js, ks-1}, {i, i-grid_w*grid_h},
                    -V[2](is, js, ks), V_solid[2](is, js, ks), A, b, scale, LS, b_index, label_in, do_tension);
            build_A_element({is, js, ks}, {is, js, ks+1}, {i, i+grid_w*grid_h},
                    V[2](is, js, ks+1), -V_solid[2](is, js, ks+1), A, b, scale, LS, b_index, label_in, do_tension);

            scalar_t diag = A.coeffRef(i,i);
            Ad.coeffRef(i,i) = 1.0 / diag;
        } else if (label_in->get_label_center(is, js, ks) == FLUID_CENTER) {
            A.coeffRef(i,i) = 1.0;
            Ad.coeffRef(i,i) = 1.0;
            b(b_index) = 0.0;
        } else {
            A.coeffRef(i,i) = 1.0;
            Ad.coeffRef(i,i) = 1.0;
            b(b_index) = 0.0;
        }
    }
}

void SimWater::pressure_gradient_update_velocity(CubeX &v, const CubeX &v_solid, const CubeX &p, const CubeX &LS, const Vector3ui &face,
        scalar_t scale, const SimLabel *label_in, bool do_tension, bool check_air){
    unsigned int d = 0;
    scalar_t MAXP = 1000;
    for (auto &val : v){
        Vector3ui I = convert_index_to_coords(d, v.n_rows, v.n_cols);
        // Use these to keep track of the index of the second cell with shared face.
        Vector3ui IF = I - face;

        if (label_in->get_label_center(I(0), I(1), I(2)) == FLUID
            || label_in->get_label_center(IF(0), IF(1), IF(2)) == FLUID
            || label_in->get_label_center(I(0), I(1), I(2)) == FLUID_CENTER
            || label_in->get_label_center(IF(0), IF(1), IF(2)) == FLUID_CENTER)
        {
            if (label_in->get_label_center(I(0), I(1), I(2)) == 2 || label_in->get_label_center(IF(0), IF(1), IF(2)) == 2){
                v(I(0), I(1), I(2)) = v_solid(I(0), I(1), I(2));
            } else {
                if(label_in->get_label_center(I(0), I(1), I(2)) == 0){
                    scalar_t ghost_p = 0;
                    if (do_tension) {
                        scalar_t theta = 0;
                        Vector3 interface = get_interface_position_between_cells(IF, I, LS, theta);
                        scalar_t curvature = lambda * simLS->get_curvature(interface);
                        ghost_p = ((theta - 1) / theta) * p(IF(0), IF(1), IF(2)) + curvature / theta;
                        ghost_p = sgn(ghost_p)*std::min(MAXP, std::abs(ghost_p));
                    }
                    v(I(0), I(1), I(2)) -= -scale*(p(IF(0), IF(1), IF(2))-ghost_p);
                } else if (label_in->get_label_center(IF(0), IF(1), IF(2)) == 0){
                    scalar_t ghost_p = 0;
                    if (do_tension) {
                        scalar_t theta = 0;
                        Vector3 interface = get_interface_position_between_cells(I, IF, LS, theta);
                        scalar_t curvature = lambda * simLS->get_curvature(interface);
                        ghost_p = ((theta - 1) / theta) * p(I(0), I(1), I(2)) + curvature / theta;
                        ghost_p = sgn(ghost_p)*std::min(MAXP, std::abs(ghost_p));
                    }
                    v(I(0), I(1), I(2)) -= -scale*(ghost_p-p(I(0), I(1), I(2)));
                } else {
                    v(I(0), I(1), I(2)) -= (scale *p(I(0), I(1), I(2)) - scale * p(IF(0), IF(1), IF(2)));
                }
            }
        }

        d++;
    }
}

void SimWater::pressure_gradient_update(const CubeX &p, scalar_t dt, CubeX& LS, const SimLabel *label_in, bool do_tension, bool check_air){
    scalar_t scale = dt / (density * scale_w); //Assuming equal scale for w and h.
//    std::cout << "Scale for gradient update " << scale << std::endl;
    pressure_gradient_update_velocity(V[0], V_solid[0], p, LS, {1, 0, 0}, scale, label_in, do_tension, check_air);
    pressure_gradient_update_velocity(V[1], V_solid[1], p, LS, {0, 1, 0}, scale, label_in, do_tension, check_air);
    pressure_gradient_update_velocity(V[2], V_solid[2], p, LS, {0, 0, 1}, scale, label_in, do_tension, check_air);
}

void SimWater::build_A_and_b_viscosity(MatrixA &A, VectorXs &b, MatrixA &Ad, unsigned int nx, unsigned int ny, unsigned int nz, Vector3 offset){
    // Basic matrix properties.
    scalar_t scale = (nu*dt)/(scale_w*scale_h);

    for (unsigned int d = 0; d < nx*ny*nz; d++){ // Walk along diagonal (d) of sparse matrix
        // Get the relevant spatial coordinates.
        Vector3ui I = convert_index_to_coords(d, nx, ny);
        Vector3 X = (I + offset)*scale_w;

        // Remember that our matrix is A=I-D, where D is a Laplacian style linear system.
        scalar_t on_diag = -6;
        if(fluid_label->get_label(X) == 1) {
            if (fluid_label->get_label(X(0) - scale_w, X(1), X(2)) == 1) {
                A.coeffRef(d, d - 1) = -scale;
            } else {
                on_diag++;
            }
            if (fluid_label->get_label(X(0) + scale_w, X(1), X(2)) == 1) {
                A.coeffRef(d, d + 1) = -scale;
            } else {
                on_diag++;
            }
            if (fluid_label->get_label(X(0), X(1) - scale_h, X(2)) == 1) {
                A.coeffRef(d, d - nx) = -scale;
            } else {
                on_diag++;
            }
            if (fluid_label->get_label(X(0), X(1) + scale_h, X(2)) == 1) {
                A.coeffRef(d, d + nx) = -scale;
            } else {
                on_diag++;
            }
            if (fluid_label->get_label(X(0), X(1), X(2) - scale_d) == 1) {
                A.coeffRef(d, d - nx*ny) = -scale;
            } else {
                on_diag++;
            }
            if (fluid_label->get_label(X(0), X(1), X(2) + scale_d) == 1) {
                A.coeffRef(d, d + nx*ny) = -scale;
            } else {
                on_diag++;
            }
            scalar_t diag = 1.0 - scale * on_diag;
//            diag = 1.0;
            A.coeffRef(d, d) = diag;
            Ad.coeffRef(d, d) = 1.0 / diag;
        } else {
            scalar_t diag = 1.0;
            A.coeffRef(d, d) = diag;
            Ad.coeffRef(d, d) = 1.0 / diag;
        }
    }
}

void SimWater::check_divergence(CubeX &grid_div){
    bool bad_divergence = false;
    for (int k = 0; k < grid_d; k++){
        for (int j = 0; j < grid_h; j++){
            for (int i = 0; i < grid_w; i++){
                if (air_label->label(i+1,j+1,k+1) == 1 || fluid_label->label(i+1,j+1,k+1) == 1 ){
                    scalar_t div = V[0](i+1, j, k) - V[0](i, j, k);
                    div += V[1](i, j+1, k) - V[1](i, j, k);
                    div += V[2](i, j, k+1) - V[2](i, j, k);
                    grid_div(i,j,k) = div;
                    if (std::abs(div) > 1E-10){
//                        std::cout << "Divergence too large at " << div << std::endl;
                        if (!bad_divergence){
                            bad_divergence = true;
                        }
                    }
                }
            }
        }
    }

    if (bad_divergence){
        std::cout << "Found a bad divergence, see divergences below." << std::endl;
        std::cout << grid_div << std::endl;
        std::cout << "Air Labels" << std::endl;
        std::cout << air_label->label << std::endl;
    }
}