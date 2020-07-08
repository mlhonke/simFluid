//
// Created by graphics on 2020-06-23.
//

#include "sim_viewer.hpp"
#include "mesh_output.hpp"

#include <thread>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/png/writePNG.h>

void get_file_name(char* filename, int max_filename, const char* prefix, int frame_id, const char* type){
    snprintf(filename, max_filename, "%s_%07d.%s", prefix, frame_id, type);
}

bool save_screen(igl::opengl::glfw::Viewer &v, int frame_id){
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R(500,500);
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G(500,500);
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B(500,500);
    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> T(500,500);
    v.core().draw_buffer(v.data(),false,R,G,B,T);

    //Setup filename
    const unsigned int max_filename = 256;
    char filename[max_filename];
    char const *prefix = "../screens/screen";
    char const *type = "png";
    get_file_name(filename, max_filename, prefix, frame_id, type);

    igl::png::writePNG(R,G,B,T,filename);

    return false;
}

void run_viewer(SimViewer &viewer){
    std::cout << "Viewer launching" << std::endl;
    viewer.getViewer()->callback_pre_draw = [&viewer](igl::opengl::glfw::Viewer & v){
        std::cout << "Viewer callback loop" << std::endl;
        std::unique_lock<std::mutex> lck(viewer.viewerMtx);
        if (!viewer.refresh) {
            viewer.viewerCV.wait(lck);
        }
        std::cout << "Viewer callback loop post CV" << std::endl;
        v.data().set_mesh(viewer.Vmesh, viewer.Fmesh);
        v.data().set_normals(viewer.Nmesh);
        Eigen::Vector3d diffuse, ambient, specular;
        diffuse = Eigen::Vector3d::Constant(0.4f);
        ambient = Eigen::Vector3d::Constant(0.1f);
        specular = Eigen::Vector3d::Constant(0.5f);
        v.data().uniform_colors(diffuse, ambient, specular);
        v.data().shininess = 2.0f;
        viewer.refresh = false; // This lets the sim thread operate on the mesh since we're done with it here.
        save_screen(v, viewer.frame_id++);
        return false;
    };
    // Set Viewer to tight draw loop
    viewer.getViewer()->core().is_animating = true;
    // Launch viewer in this thread
    viewer.getViewer()->launch(false, false, "Simulation Render", 500, 500);
}

SimViewer::SimViewer(SimParams &C):C(C) {
    iglviewer = new igl::opengl::glfw::Viewer;

    iglviewer->core().camera_dnear = 0.1f;
    iglviewer->core().camera_dfar = 100.0f;
    iglviewer->core().camera_view_angle = 45.0f;
    // Sim data may be in double types, so explicitly cast to float so compiler knows this is intended.
    Eigen::Matrix<scalar_t, 3, 1> camera_eye = {C.sim_w/2.0, -0.75*C.sim_w, 2.0*C.sim_d};
    iglviewer->core().camera_eye = camera_eye.cast<float>();
    Eigen::Matrix<scalar_t, 3, 1> camera_center =  {C.sim_w/2.0, C.sim_w/2.0, C.sim_d/2.0};
    iglviewer->core().camera_center = camera_center.cast<float>();
    iglviewer->core().camera_up = {0, 1, 0};
    iglviewer->core().unset(iglviewer->data().show_lines);
    Eigen::Matrix<scalar_t, 3, 1> light_position = {C.sim_w/2.0, 0.5*C.sim_w, 2.0*C.sim_d};
    iglviewer->core().light_position = light_position.cast<float>();
}

void SimViewer::refresh_viewer() {
    std::cout << "Refreshing viewer" << std::endl;
    iglviewer->data().clear();
    refresh = true;
    viewerCV.notify_one();
    viewerMtx.unlock();
}

void SimViewer::wait_for_viewer() {
    // if lock is held, then viewer is still drawing to screen, simulation has to pause so that the current frame
    // can be rendered. Or else it could corrupt viewer data if it tries to alter the viewer's mesh while drawing!
    viewerMtx.lock();
}

void SimViewer::launch(){
    std::thread viewThread (run_viewer, std::ref(*this));
    viewThread.detach();
}

igl::opengl::glfw::Viewer* SimViewer::getViewer(){
    return iglviewer;
}

void SimViewer::write_mesh_to_ply() {
    //Setup filename
    const unsigned int max_filename = 256;
    char filename[max_filename];
    char const *prefix = "../screens/mesh";
    char const *type = "ply";
    get_file_name(filename, max_filename, prefix, frame_id, type);

    export_ply_mesh(filename, Vmesh, Nmesh, Fmesh);
}
