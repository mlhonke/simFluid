//
// Created by graphics on 2020-06-23.
//

#ifndef SIM_VIEWER_HPP
#define SIM_VIEWER_HPP

#include <mutex>
#include <condition_variable>

#include "eigen_types.hpp"
#include "sim_params.hpp"

namespace igl {namespace opengl {namespace glfw {class Viewer;}}}

class SimViewer {
public:
    SimViewer(SimParams &C);
    void launch();
    igl::opengl::glfw::Viewer* getViewer();
    void refresh_viewer();
    void wait_for_viewer();
    void write_mesh_to_ply();

    std::mutex viewerMtx;
    std::condition_variable viewerCV;
    bool refresh = false;
    Eigen::MatrixXi Fmesh;
    Eigen::MatrixXd Vmesh;
    Eigen::MatrixXd Nmesh;
    int frame_id = 0;

private:
    SimParams &C;
    igl::opengl::glfw::Viewer* iglviewer;
};


#endif //SIM_VIEWER_HPP
