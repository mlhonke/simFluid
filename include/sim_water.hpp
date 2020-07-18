#ifndef SIM_WATER_HPP
#define SIM_WATER_HPP

#include "sim.hpp"
#include "recorder.h"
#include "sim_params_water.hpp"
#include "eigen_types.hpp"
#include "execTimer.hpp"

class CudaCG;
class SimLevelSet;
class SimLabel;
class SimViewer;
namespace marchingtets { class MarchingTets; }
namespace igl {namespace opengl {namespace glfw { class Viewer; }}}

class SimWater : public Sim {
public:
    SimWater(SimParams C, SimWaterParams CW);
    static void create_params_from_args(int argc, char** argv, int& n_steps, SimParams *&retC, SimWaterParams *&retCW);
    static void create_params_from_args(int argc, char** argv, int& n_steps, SimParams *&retC, SimWaterParams *&retCW, int &i);
    void update_labels_from_level_set();
    void step();
    void save_data();
    void load_data();
    void initialize_fluid(SimLevelSet *level_set);
    SimLevelSet *simLS;
    bool write_mesh = false;

protected:
    void add_gravity_to_velocity(CubeX &v, scalar_t dt);
    void pressure_gradient_update_velocity(CubeX &v, const CubeX &v_solid, const CubeX &p, const CubeX &LS,
            const Vector3ui &face, scalar_t scale, const SimLabel *label_in, bool do_tension, bool check_air);
    void pressure_gradient_update(const CubeX &p, scalar_t dt, CubeX& LS, const SimLabel *label_in, bool do_tension = true, bool check_air = true);
    Vector3 get_interface_position_between_cells(const Vector3ui &wet, const Vector3ui &dry, const CubeX &LS, scalar_t &theta);
    void build_A_element(const Vector3ui &ISC, const Vector3ui &IS, const Vector2ui &IA, scalar_t vel,
            scalar_t vel_solid, MatrixA &A, VectorXs &b, scalar_t scale, const CubeX &LS, unsigned int b_index,
            const SimLabel *label_in, bool do_tension);
    void build_A_and_b(MatrixA &A, VectorXs &b, scalar_t dt, const CubeX &LS, scalar_t c, const SimLabel *label_in, bool do_tension, bool check_air);
    void build_A_and_b_viscosity(MatrixA &A, VectorXs &b, MatrixA &Ad, unsigned int nx, unsigned int ny, unsigned int nz, Vector3 offset);
    void extrapolate_velocities_from_LS();
    void extrapolate_velocity_from_LS(CubeX &v, Vector3i face);
    void set_boundary_velocities();
    void solve_pressure(const SimLabel *labels_in, bool do_tension, bool do_vol_correct, bool check_air);
    void solve_viscosity_vel(MatrixA &A_vis, MatrixA &A_visd, VectorXs &vel_new, CubeX &vel, const Vector3 &offset);
    void solve_viscosity();
    void update_triangle_mesh();
    void advect_velocity();
    void check_divergence(CubeX &grid_div);
    void update_labels_for_air();
    void update_viewer_triangle_mesh();

    int n_cells_use;
    marchingtets::MarchingTets* tets;
    VectorXs p;
    CubeX P;
    MatrixA A;
    MatrixA Ad;
    VectorXs b;
    scalar_t eps = 1E-10;
    MatrixA A_vis_u;
    MatrixA A_vis_v;
    MatrixA A_vis_w;
    MatrixA A_vis_ud;
    MatrixA A_vis_vd;
    MatrixA A_vis_wd;
    VectorXs u_new;
    VectorXs v_new;
    VectorXs w_new;
    bool run_water_unit_tests();
    scalar_t volume_old;
    scalar_t mass_controller = 1.0;
    CudaCG* cudacg_water;
    CudaCG* cudacg_vis;
    SimLabel* air_label;
    SimViewer* sim_viewer;

    //Physical constants
    scalar_t const g = -9.8;
    scalar_t const density = 1000.0;
    scalar_t const lambda = 0.2;
    scalar_t const nu = 0.1;
};

#endif
