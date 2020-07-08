//
// Created by graphics on 2020-07-07.
//

#include <fstream>
#include <string>

#include "mesh_output.hpp"

void output_ply_header_to_file(std::ofstream &mesh_file, unsigned int n_vertices, unsigned int n_faces, bool normals = false){
    mesh_file << "ply\nformat ascii 1.0" << std::endl;
    mesh_file << "element vertex " << n_vertices << std::endl;
    mesh_file << "property float x\nproperty float y\nproperty float z" << std::endl;
    if (normals)
        mesh_file << "property float nx\nproperty float ny\nproperty float nz" << std::endl;
    mesh_file << "element face " << n_faces << std::endl;
    mesh_file << "property list uchar int vertex_indices" << std::endl;
    mesh_file << "end_header" << std::endl;
}

void export_ply_mesh(const char* filename, Eigen::MatrixXd &Vmesh, Eigen::MatrixXd &Nmesh, Eigen::MatrixXi &Fmesh){
    std::ofstream mesh_file;
    mesh_file.open(filename);
    output_ply_header_to_file(mesh_file, Vmesh.rows(), Fmesh.rows(), true);

    for (int i = 0; i < Vmesh.rows(); i++){
        mesh_file << Vmesh(i,0) << " " << Vmesh(i,1) << " " << Vmesh(i,2) << " ";
        mesh_file << Nmesh(i,0) << " " << Nmesh(i,1) << " " << Nmesh(i,2) << std::endl;
    }

    for (int i = 0; i < Fmesh.rows(); i++){
        mesh_file << "3 " << Fmesh(i,0) << " " << Fmesh(i,1) << " " << Fmesh(i,2) << std::endl;
    }

    mesh_file.close();
}

void export_ply_mesh(const char* filename, std::vector<Vector3> &Vmesh, std::vector<Vector3i> &Fmesh){
    std::ofstream mesh_file;
    mesh_file.open(filename);
    output_ply_header_to_file(mesh_file, Vmesh.size(), Fmesh.size());

    for (auto & V : Vmesh){
        mesh_file << V(0) << " " << V(1) << " " << V(2) << " " << std::endl;
    }

    for (auto & F : Fmesh){
        mesh_file << "3 " << F(0) << " " << F(1) << " " << F(2) << " " << std::endl;
    }

    mesh_file.close();
}