//
// Created by graphics on 27/01/19.
//
#include "sim_water.hpp"
#include "fd_math.hpp"
#include <vector>

bool SimWater::run_water_unit_tests() {
//    MatrixA A_test((grid_w+1)*grid_h, (grid_w+1)*grid_h);
//    VectorXs b_test = VectorXs::Constant(grid_w*(grid_h+1), 1, 1);
//    build_A_and_b_viscosity(A_test, b_test, grid_w+1, grid_h, -0.5, 0);
//#ifdef DEBUGVIS
//    std::cout << A_test << std::endl;
//    std::cout << b_test << std::endl;
//#endif

// Test polygon area calculation.
//    std::vector<std::vector<Vector2>> polys;
//    std::vector<Vector2> poly, poly2;
//    poly.push_back(Vector2(1, 1));
//    poly.push_back(Vector2(-1, 1));
//    poly.push_back(Vector2(-1, -1));
//    poly.push_back(Vector2(1, -1));
//    poly.push_back(Vector2(1, 1));
//    polys.push_back(poly);
//
//    poly2.push_back(Vector2(1, 1));
//    poly2.push_back(Vector2(2, 1));
//    poly2.push_back(Vector2(2, 2));
//    poly2.push_back(Vector2(1, 1));
//    polys.push_back(poly2);
//
//    std::cout << "Testing area calculation for polygons" << std::endl;
//    std::cout << calc_area_polygons(polys) << std::endl;

//    std::cout << "Testing RK3 Advection" << std::endl;
//    CubeX Q(grid_w, grid_h, grid_d, arma::fill::zeros);
//    Q(grid_w/2, grid_h/2, grid_d/2) = 1;
//    advect_RK3(Q, {0,0,0}, V, label, dt, true, false, true);
//    std::cout << Q << std::endl;
    return true;
}
