#ifndef SIMCOMMON_ADVECT_HPP
#define SIMCOMMON_ADVECT_HPP

#include "sim_external_types.hpp"
#include "sim_params.hpp"

void advect_RK2(CubeX &q, const Vector3 &offset, std::array<CubeX, 3> &V, scalar_t dt, SimParams &C, bool do_clamp, bool do_clamp_q);
void advect_RK3(CubeX &q, const Vector3 &offset, std::array<CubeX, 3> &V, scalar_t dt, SimParams &C, bool do_clamp, bool do_clamp_q);
void advect_RK3_CUDA(CubeX &q, const Vector3 &offset, std::array<scalar_t*, 3> DEV_V, scalar_t dt, SimParams* DEV_C,
                     bool do_clamp = false, bool do_clamp_q = false);
Vector3 advect_particle_RK3(const Vector3 &X, std::array<CubeX, 3> &V, scalar_t dt, SimParams &C);

Vector3 lerp_velocity(const Vector3 &X, const macVel &V, scalar_t dx, bool set_bounds = false);
Vector3 cerp_velocity(const Vector3 &X, const macVel &V, scalar_t dx, bool set_bounds = true);


#endif //SIMCOMMON_ADVECT_HPP
