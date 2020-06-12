//
// Created by graphics on 24/09/19.
//

#ifndef FERRO3D_SIM_LABEL_HPP
#define FERRO3D_SIM_LABEL_HPP

#include "sim_params_water.hpp"
#include "sim.hpp"
#include <queue>
#include <string>

class SimLabel {
public:
    SimLabel(Sim &sim, bool init_colours);

    int get_label(Vector3 X, bool tell_me_boundary = false) const;
    int get_label(scalar_t x, scalar_t y, scalar_t z, bool tell_me_boundary = false) const;
    int get_label_center(int i, int j, int k) const;
    int get_label_center(const Vector3ui &I) const;
    int get_cell_count(int cell_type) const;
    void save_data(std::string filename);
    void load_data(std::string filename);

    void colour_cell(const Vector3i &coord, const Vector3i &face, const CubeX &LS, std::queue<Vector3i> &search,
                     int new_colour, std::vector<scalar_t> &best_ls);
    void colour_label(const CubeX &LS);

    void update_label_on_device();
    int* DEV_label;
    CubeXi label;
    CubeXi colours;
    std::vector<bool> has_air; // confusing nomenclature since we're treating air as a fluid.
    std::vector<Vector3i> centers;

private:
    bool use_colours = false;
    Sim& sim;
    const int grid_w;
    const int grid_h;
    const int grid_d;
    const scalar_t dx;
};


#endif //FERRO3D_SIM_LABEL_HPP
