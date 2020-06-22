#ifndef INTERPOLATE_HPP
#define INTERPOLATE_HPP

#include "sim_external_types.hpp"

// Non SIM class member interpolation functions
scalar_t lerp(scalar_t x, scalar_t x1, scalar_t x2, scalar_t Q1, scalar_t Q2);
scalar_t grid_trilerp(const Vector3 &X, const CubeX &q, scalar_t const dx);
scalar_t grid_bilerp(scalar_t x, scalar_t y, const MatrixX &q, scalar_t const dx);
scalar_t grid_bilerp(const Vector2 &pos, const MatrixX &q, scalar_t const dx);
scalar_t bilerp(scalar_t x, scalar_t y, scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2, const Matrix2 &Q);
scalar_t harmonic_mean_interpolate(scalar_t a, scalar_t b);
scalar_t grid_tricerp(const Vector3 &X, const CubeX &q, scalar_t const dx, bool clamp = true);
scalar_t grid_bicerp(scalar_t x, scalar_t y, const MatrixX &q, scalar_t const dx, bool clamp = true);
scalar_t bicerp(scalar_t x, scalar_t y, Vector4 X, Vector4 Y, Matrix4 &Q, bool clamp);
scalar_t clamp4(scalar_t a, Matrix2 vals);
scalar_t clamp(scalar_t a, scalar_t min, scalar_t max);
scalar_t cerp(scalar_t x, const Vector4 &X, const Vector4 &Q, bool clamp);
scalar_t biclamp(scalar_t val, scalar_t max);

template <typename T> int sgn(T val) {
    return (val > 0) - (val < 0);
}

template <typename T> int bisgn(T val) {
    return (val >= 0) - (val < 0);
}
#endif
