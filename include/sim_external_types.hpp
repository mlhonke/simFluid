/* sim_external_types.hpp
 * Keep all external typedefs. Want to keep these separate from standard C ones, so that CUDA running code does not have
 * to include ARMADILLO or other external libraries (that may be incompatible with CUDA).
 */

#ifndef SIMCOMMON_SIM_ARMA_TYPES_HPP
#define SIMCOMMON_SIM_ARMA_TYPES_HPP

#include <armadillo>
#include "sim_types.hpp"

namespace Ferro {
// Vector types
    typedef arma::Col<scalar_t> VectorX;
    typedef arma::Col<int> VectorXi;
    typedef arma::Col<scalar_t>::fixed<4> Vector4;
    typedef arma::Col<scalar_t>::fixed<3> Vector3;
    typedef arma::Col<int>::fixed<3> Vector3i;
    typedef arma::Col<unsigned int>::fixed<3> Vector3ui;
    typedef arma::Col<scalar_t>::fixed<2> Vector2;
    typedef arma::Col<int>::fixed<2> Vector2i;
    typedef arma::Col<unsigned int>::fixed<2> Vector2ui;

// Matrix types
    typedef arma::Mat<scalar_t>::fixed<2, 2> Matrix2;
    typedef arma::Mat<scalar_t>::fixed<3, 3> Matrix3;
    typedef arma::Mat<scalar_t>::fixed<4, 4> Matrix4;
    typedef arma::Mat<scalar_t> MatrixX;
    typedef arma::Mat<int> MatrixXi;

// Cube types
    typedef arma::Cube<scalar_t> CubeX;
    typedef arma::Cube<int> CubeXi;
}

using namespace Ferro;

// Deep types
typedef std::array<CubeX, 3> macVel;
typedef std::array<scalar_t*, 3> DEV_macVel;

#endif //SIMCOMMON_SIM_ARMA_TYPES_HPP
