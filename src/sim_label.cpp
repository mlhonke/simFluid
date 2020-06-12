//
// Created by graphics on 24/09/19.
//

#include "sim_label.hpp"
#include <cuda_runtime.h>
#include "cuda_errorcheck.hpp"

SimLabel::SimLabel(Sim& sim, bool init_colours):sim(sim), grid_w(sim.grid_w), grid_h(sim.grid_h), grid_d(sim.grid_d), dx(sim.scale_w){
    label = CubeXi(grid_w+2, grid_h+2, grid_d+2, arma::fill::zeros);
    cuda_check(cudaMalloc(&DEV_label, label.n_elem*sizeof(int)))

    use_colours = init_colours;
    if (init_colours){
        colours = CubeXi(grid_w, grid_h, grid_d);
    }
}

void SimLabel::colour_cell(const Vector3i &coord, const Vector3i &face, const CubeX &LS,
    std::queue<Vector3i> &search, int new_colour, std::vector<scalar_t> &best_ls){
    Vector3i n_coord = coord + face;
    if (get_label_center(n_coord(0), n_coord(1), n_coord(2)) == 1
        && colours(n_coord(0), n_coord(1), n_coord(2)) == -1){
        colours(n_coord(0), n_coord(1), n_coord(2)) = new_colour;
        scalar_t cur_LS = std::abs(LS(n_coord(0), n_coord(1), n_coord(2)));
        if (cur_LS > best_ls[new_colour]){
            best_ls[new_colour] = cur_LS;
            centers[new_colour] = n_coord;
        }
        search.push(n_coord);
    } else if (get_label_center(n_coord(0), n_coord(1), n_coord(2)) == 0){
        has_air[new_colour] = true;
    }
}

void SimLabel::colour_label(const CubeX &LS){
    int ni = label.n_rows-2;
    int nj = label.n_cols-2;
    int nk = label.n_slices-2;
    colours.fill(-1);
    has_air.clear();
    centers.clear();
    int new_colour = -1;
    std::queue<Vector3i> search;
    std::vector<scalar_t> best_ls; // greatest abs(LS) value gives the center of a fluid segment.

    for (int k = 0; k < nk; k++){
        for (int j = 0; j < nj; j++){
            for (int i = 0; i < ni; i++){
                if (get_label_center(i,j,k) == 1 && colours(i,j,k) == -1){
                    new_colour++;
                    centers.push_back({i,j,k}); // speculative, update later with correct center.
                    best_ls.push_back(std::abs(LS(i,j,k)));
                    has_air.push_back(false); // assume false unless we find connected air.
                    colours(i,j,k) = new_colour;
                    search.push({i,j,k});
                    while (!search.empty()){
                        Vector3i coord = search.front();
                        search.pop();
                        colour_cell(coord, {1, 0, 0}, LS, search, new_colour, best_ls);
                        colour_cell(coord, {-1, 0, 0}, LS, search, new_colour, best_ls);
                        colour_cell(coord, {0, 1, 0}, LS, search, new_colour, best_ls);
                        colour_cell(coord, {0, -1, 0}, LS, search, new_colour, best_ls);
                        colour_cell(coord, {0, 0, 1}, LS, search, new_colour, best_ls);
                        colour_cell(coord, {0, 0, -1}, LS, search, new_colour, best_ls);
                    }
                }
            }
        }
    }

    // Mark non-air containing fluid center points with special label for use in pressure solve.
    int i = 0;
    for (const auto& center : centers){
        if (has_air[i] == false){
            label(center(0)+1, center(1)+1, center(2)+1) = FLUID_CENTER;
            std::cout << "Saved the simulation" << std::endl;
        }
        i++;
    }
}

void SimLabel::update_label_on_device(){
    cuda_check(cudaMemcpy(DEV_label, label.memptr(), label.n_elem*sizeof(int), cudaMemcpyHostToDevice));
}

int SimLabel::get_label_center(int i, int j, int k) const {
    return label(i+1, j+1, k+1);
}

int SimLabel::get_label_center(const Vector3ui &I) const {
    return get_label_center(I(0), I(1), I(2));
}

int SimLabel::get_label(Vector3 X, bool tell_me_boundary) const {
    return get_label(X(0), X(1), X(2), tell_me_boundary);
}

int SimLabel::get_label(scalar_t x, scalar_t y, scalar_t z, bool tell_me_boundary) const {
    scalar_t eps = 0.000001;
    scalar_t ipart = x * (1/dx);
    scalar_t jpart = y * (1/dx);
    scalar_t kpart = z * (1/dx);
    auto i = (int) roundf(ipart);
    auto j = (int) roundf(jpart);
    auto k = (int) roundf(kpart);
    int il = i+1;
    int jl = j+1;
    int kl = k+1;
    ipart -= i;
    jpart -= j;
    kpart -= k;

    if (il >= 0 && il < grid_w+2 && jl >= 0 && jl < grid_h+2 && kl >= 0 && kl < grid_d+2) {
        int cell1 = label(il, jl, kl);
        int cell2 = -1; // -1 for not on an edge.
        if (std::abs(ipart - 0.5) < eps) {
            cell2 = label(il + 1, jl, kl);
        } else if (std::abs(ipart + 0.5) < eps) {
            cell2 = label(il - 1, jl, kl);
        } else if (std::abs(jpart - 0.5) < eps) {
            cell2 = label(il, jl+1, kl);
        } else if (std::abs(jpart + 0.5) < eps) {
            cell2 = label(il, jl-1, kl);
        } else if (std::abs(kpart - 0.5) < eps) {
            cell2 = label(il, jl, kl+1);
        } else if (std::abs(kpart + 0.5) < eps) {
            cell2 = label(il, jl, kl-1);
        }

        if (tell_me_boundary){
            if ((cell1 == 0 && cell2 == 1) || (cell1 == 1 && cell2 == 0)){ //Air fluid boundary
                return 10;
            }
            if ((cell1 == 1 && cell2 == 2) || (cell1 == 2 && cell2 == 1)){ //Fluid solid boundary
                return 12;
            }
        }

        if ( cell1 == 2 || cell2 == 2 ){
            return 2;
        } else if ( cell1 == 1 || cell2 == 1){
            return 1;
        } else {
            return 0;
        }

    } else {
        return 2; //TODO: Figure out a better way of saying "invalid points" maybe with a bool return or something.
    }
}

int SimLabel::get_cell_count(int cell_type) const {
    int n_cell = 0;
    for (unsigned int k = 0; k < grid_d+2; k++) {
        for (unsigned int j = 0; j < grid_h+2; j++) {
            for (unsigned int i = 0; i < grid_w+2; i++) {
                if (label(i,j,k) == cell_type){
                    n_cell++;
                }
            }
        }
    }
    return n_cell;
}

void SimLabel::save_data(std::string filename){
    label.save(filename);
}

void SimLabel::load_data(std::string filename){
    label.load(filename);
    update_label_on_device();
}