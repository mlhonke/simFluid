#include "marchingtets.h"

namespace marchingtets {

// add triangles for contour in the given cube (from grid points (i,j,k) to (i+1,j+1,k+1)) if not there already
    void MarchingTets::
    contour_cube(int i, int j, int k) {
        if (cube_record.has_entry(Vec3i(i, j, k))) // if we have done this grid cell before
            return; // then there's nothing to do here

        Vec3i x000(i, j, k), x001(i, j, k + 1),
                x010(i, j + 1, k), x011(i, j + 1, k + 1),
                x100(i + 1, j, k), x101(i + 1, j, k + 1),
                x110(i + 1, j + 1, k), x111(i + 1, j + 1, k + 1);
        float p000 = eval(i, j, k), p001 = eval(i, j, k + 1),
                p010 = eval(i, j + 1, k), p011 = eval(i, j + 1, k + 1),
                p100 = eval(i + 1, j, k), p101 = eval(i + 1, j, k + 1),
                p110 = eval(i + 1, j + 1, k), p111 = eval(i + 1, j + 1, k + 1);

        if ((i + j + k) % 2) {
            contour_tet(x000, x110, x011, x010, p000, p110, p011, p010);
            contour_tet(x000, x101, x110, x100, p000, p101, p110, p100);
            contour_tet(x101, x011, x110, x111, p101, p011, p110, p111);
            contour_tet(x000, x011, x101, x001, p000, p011, p101, p001);
            contour_tet(x000, x110, x101, x011, p000, p110, p101, p011);
        } else {
            contour_tet(x100, x010, x001, x000, p100, p010, p001, p000);
            contour_tet(x100, x111, x010, x110, p100, p111, p010, p110);
            contour_tet(x100, x001, x111, x101, p100, p001, p111, p101);
            contour_tet(x001, x010, x111, x011, p001, p010, p111, p011);
            contour_tet(x100, x010, x111, x001, p100, p010, p111, p001);
        }

        cube_record.add(Vec3i(i, j, k), 1); // and mark that we're now down this cube
    }

// contour the tet with given grid point vertices and function values
// --- corners arranged so that 0-1-2 uses right-hand-rule to get to 3
    void MarchingTets::
    contour_tet(const Vec3i &x0, Vec3i &x1, Vec3i &x2, Vec3i &x3, float p0, float p1, float p2, float p3) {
        // guard against topological degeneracies
        if (p0 == 0) p0 = 1e-30f;
        if (p1 == 0) p1 = 1e-30f;
        if (p2 == 0) p2 = 1e-30f;
        if (p3 == 0) p3 = 1e-30f;

        if (p0 < 0) {
            if (p1 < 0) {
                if (p2 < 0) {
                    if (p3 < 0) {
                        return; // no contour here
                    } else // p3>=0
                        tri.push_back({find_edge_cross(x0, x3, p0, p3),
                                       find_edge_cross(x1, x3, p1, p3),
                                       find_edge_cross(x2, x3, p2, p3)});
                } else { // p2>=0
                    if (p3 < 0)
                        tri.push_back({find_edge_cross(x0, x2, p0, p2),
                                       find_edge_cross(x3, x2, p3, p2),
                                       find_edge_cross(x1, x2, p1, p2)});
                    else { // p3>=0
                        tri.push_back({find_edge_cross(x0, x3, p0, p3),
                                       find_edge_cross(x1, x3, p1, p3),
                                       find_edge_cross(x0, x2, p0, p2)});
                        tri.push_back({find_edge_cross(x1, x3, p1, p3),
                                       find_edge_cross(x1, x2, p1, p2),
                                       find_edge_cross(x0, x2, p0, p2)});
                    }
                }
            } else { // p1>=0
                if (p2 < 0) {
                    if (p3 < 0)
                        tri.push_back({find_edge_cross(x0, x1, p0, p1),
                                       find_edge_cross(x2, x1, p2, p1),
                                       find_edge_cross(x3, x1, p3, p1)});
                    else { // p3>=0
                        tri.push_back({find_edge_cross(x0, x3, p0, p3),
                                       find_edge_cross(x0, x1, p0, p1),
                                       find_edge_cross(x2, x3, p2, p3)});
                        tri.push_back({find_edge_cross(x0, x1, p0, p1),
                                       find_edge_cross(x2, x1, p2, p1),
                                       find_edge_cross(x2, x3, p2, p3)});
                    }
                } else { // p2>=0
                    if (p3 < 0) {
                        tri.push_back({find_edge_cross(x0, x1, p0, p1),
                                       find_edge_cross(x0, x2, p0, p2),
                                       find_edge_cross(x3, x2, p3, p2)});
                        tri.push_back({find_edge_cross(x0, x1, p0, p1),
                                       find_edge_cross(x3, x2, p3, p2),
                                       find_edge_cross(x3, x1, p3, p1)});
                    } else // p3>=_0
                        tri.push_back({find_edge_cross(x0, x1, p0, p1),
                                       find_edge_cross(x0, x2, p0, p2),
                                       find_edge_cross(x0, x3, p0, p3)});
                }
            }
        } else { // p0>=0
            if (p1 < 0) {
                if (p2 < 0) {
                    if (p3 < 0)
                        tri.push_back({find_edge_cross(x0, x1, p0, p1),
                                       find_edge_cross(x0, x3, p0, p3),
                                       find_edge_cross(x0, x2, p0, p2)});
                    else { // p3>=0
                        tri.push_back({find_edge_cross(x0, x1, p0, p1),
                                       find_edge_cross(x3, x1, p3, p1),
                                       find_edge_cross(x3, x2, p3, p2)});
                        tri.push_back({find_edge_cross(x3, x2, p3, p2),
                                       find_edge_cross(x0, x2, p0, p2),
                                       find_edge_cross(x0, x1, p0, p1)});
                    }
                } else { // p2>=0
                    if (p3 < 0) {
                        tri.push_back({find_edge_cross(x0, x1, p0, p1),
                                       find_edge_cross(x0, x3, p0, p3),
                                       find_edge_cross(x3, x2, p3, p2)});
                        tri.push_back({find_edge_cross(x0, x1, p0, p1),
                                       find_edge_cross(x3, x2, p3, p2),
                                       find_edge_cross(x2, x1, p2, p1)});
                    } else // p3>=0
                        tri.push_back({find_edge_cross(x1, x0, p1, p0),
                                       find_edge_cross(x1, x3, p1, p3),
                                       find_edge_cross(x1, x2, p1, p2)});
                }
            } else { // p1>=0
                if (p2 < 0) {
                    if (p3 < 0) {
                        tri.push_back({find_edge_cross(x1, x3, p1, p3),
                                       find_edge_cross(x0, x3, p0, p3),
                                       find_edge_cross(x0, x2, p0, p2)});
                        tri.push_back({find_edge_cross(x1, x3, p1, p3),
                                       find_edge_cross(x0, x2, p0, p2),
                                       find_edge_cross(x1, x2, p1, p2)});
                    } else // p3>=0
                        tri.push_back({find_edge_cross(x0, x2, p0, p2),
                                       find_edge_cross(x1, x2, p1, p2),
                                       find_edge_cross(x3, x2, p3, p2)});
                } else { // p2>=0
                    if (p3 < 0)
                        tri.push_back({find_edge_cross(x0, x3, p0, p3),
                                       find_edge_cross(x2, x3, p2, p3),
                                       find_edge_cross(x1, x3, p1, p3)});
                    else { // p3>=0
                        return; // assume no degenerate cases (where some of the p's are zero)
                    }
                }
            }
        }
    }

// return the vertex of the edge crossing (create it if necessary) between given grid points and function values
    unsigned int MarchingTets::
    find_edge_cross(const Vec3i &x0, const Vec3i &x1, float p0, float p1) {
        unsigned int vertex_index;
        if (edge_cross.get_entry(Vec6i(x0.v[0], x0.v[1], x0.v[2], x1.v[0], x1.v[1], x1.v[2]), vertex_index)) {
            return vertex_index;
        } else if (edge_cross.get_entry(Vec6i(x1.v[0], x1.v[1], x1.v[2], x0.v[0], x0.v[1], x0.v[2]), vertex_index)) {
            return vertex_index;
        } else {
            float a = p1 / (p1 - p0), b = 1 - a;
            vertex_index = (unsigned int) x.size();
            x.push_back(origin + dx * Vector3({a * x0[0] + b * x1[0], a * x0[1] + b * x1[1], a * x0[2] + b * x1[2]}));
            edge_cross.add(Vec6i(x0.v[0], x0.v[1], x0.v[2], x1.v[0], x1.v[1], x1.v[2]), vertex_index);
            return vertex_index;
        }
    }

}