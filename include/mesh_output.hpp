//
// Created by graphics on 2020-07-07.
//

#ifndef STAIRS_MESH_OUTPUT_HPP
#define STAIRS_MESH_OUTPUT_HPP

#include "eigen_types.hpp"
#include "sim_external_types.hpp"

void export_ply_mesh(const char* filename, Eigen::MatrixXd &Vmesh, Eigen::MatrixXd &Nmesh, Eigen::MatrixXi &Fmesh);
void export_ply_mesh(const char* filename, std::vector<Vector3> &Vmesh, std::vector<Vector3i> &Fmesh);

#endif //STAIRS_MESH_OUTPUT_HPP
