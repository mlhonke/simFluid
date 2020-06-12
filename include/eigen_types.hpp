//
// Created by graphics on 30/06/19.
//

#ifndef FERRO3D_EIGEN_TYPES_HPP
#define FERRO3D_EIGEN_TYPES_HPP

#include "sim_params.hpp"
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "Eigen/IterativeLinearSolvers"

//Typedefs
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
typedef Eigen::SparseMatrix<scalar_t, Eigen::RowMajor> MatrixA;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, 1> VectorXs;

#endif //FERRO3D_EIGEN_TYPES_HPP
