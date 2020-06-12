#ifndef MARCHINGTETS_H
#define MARCHINGTETS_H

#include "hashtable.h"
#include "vec.h"
#include "sim_params.hpp"

namespace marchingtets {

    class MarchingTets {
    public:
        // the mesh that marching tets is creating
        std::vector<Vector3ui> tri; // mesh connectivity
        std::vector<Vector3> x; // mesh vertex locations

        // marching tets grid definition
        Vector3 origin; // where the grid point (0,0,0) is
        float dx; // the grid spacing

        // internal tables for saving information on what has been done so far
        HashTable<Vec6i, unsigned int> edge_cross; // stores vertices that have been created already at given edge crossings
        HashTable<Vec3i, char> cube_record; // indicates if a cube has already been contoured or not
        CubeX const *levelset;

        explicit MarchingTets(float dx_ = 1)
                : origin({0, 0, 0}), dx(dx_) {}

        explicit MarchingTets(CubeX const *levelset_, const Vector3 &origin_, float dx_ = 1)
                : levelset(levelset_), origin(origin_), dx(dx_) {}

        virtual ~MarchingTets(void) {}

        // add triangles for contour in the given cube (from grid points (i,j,k) to (i+1,j+1,k+1)) if not there already
        void contour_cube(int i, int j, int k);

        // how we actually find the value of the implicit surface function at a grid point: override this
        float eval(int i, int j, int k) {
            return (float) (*levelset)(i, j, k);
        }

        // helper functions
    private:
        void contour_tet(const Vec3i &x0, Vec3i &x1, Vec3i &x2, Vec3i &x3, float p0, float p1, float p2, float p3);

        unsigned int find_edge_cross(const Vec3i &x0, const Vec3i &x1, float p0, float p1);
    };
}

#endif
