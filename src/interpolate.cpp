#include "interpolate.hpp"

scalar_t harmonic_mean_interpolate(scalar_t a, scalar_t b){
    // Implemented from Dr. Afkhami Motion of Ferrofluid droplets after equation 3.6
    return 1/( 0.5 * (1/a + 1/b) );
}

scalar_t grid_trilerp(const Vector3 &X, const CubeX &q, scalar_t const dx){
    int k = (int) std::floor(X(2) * (1/dx));
    int nz = (int) q.n_slices;

    if (k < 0){
        k = 0;
    } else if (k > nz-2){
        k = nz-2;
    }

    scalar_t L1 = grid_bilerp(X(0), X(1), q.slice(k), dx); // Square one
    scalar_t L2 = grid_bilerp(X(0), X(1), q.slice(k+1), dx); // Square two
    scalar_t L = lerp(X(2), k*dx, (k+1)*dx, L1, L2); // Interpolate between one and two

    return L;
}

scalar_t grid_bilerp(const Vector2 &pos, const MatrixX &q, scalar_t const dx){
    return grid_bilerp(pos(0), pos(1), q, dx);
}

scalar_t grid_bilerp(scalar_t x, scalar_t y, const MatrixX &q, scalar_t const dx){
    int i = (int) std::floor(x * (1/dx));
    int j = (int) std::floor(y * (1/dx));
    int nx = (int) q.n_rows;
    int ny = (int) q.n_cols;

    if (i < 0){
        i = 0;
    } else if (i > nx-2){
        i = nx-2;
    }

    if (j < 0){
        j = 0;
    } else if (j > ny-2){
        j = ny-2;
    }

    scalar_t x1 = i * dx;
    scalar_t y1 = j * dx;
    scalar_t x2 = (i+1) * dx;
    scalar_t y2 = (j+1) * dx;

    Matrix2 Q;
    Q =     {   {q(i, j),   q(i, j+1)},
                {q(i+1, j), q(i+1, j+1)}    };
    return bilerp(x, y, x1, y1, x2, y2, Q);
}

scalar_t grid_tricerp(const Vector3 &X, const CubeX &q, scalar_t const dx, bool clamp){
    int k = (int) (std::floor(X(2) * (1.0/dx)) + 0.1);
    int nz = (int) q.n_slices;
    Vector4 L;
    Vector4 Z;

    if (k > 0 && k < nz-2){
        int i = 0;
        for (int l = k-1; l <= k+2; l++) {
            L(i) = grid_bicerp(X(0), X(1), q.slice(l), dx, clamp);
            Z(i) = l*dx;
            i++;
        }

        return cerp(X(2), Z, L, clamp);
    } else {
        return grid_trilerp(X, q, dx);
    }

}

scalar_t grid_bicerp(scalar_t x, scalar_t y, const MatrixX &q, scalar_t const dx, bool clamp){
    scalar_t retval;
    int i = (int) (x * (1/dx));
    int j = (int) (y * (1/dx));
    int nx = q.n_rows;
    int ny = q.n_cols;

    if (i > 0 && j > 0 && i < nx-2 && j < ny-2){
        Vector4 X;
        X(0) = (i-1) * dx;
        X(1) = i * dx;
        X(2) = (i+1) * dx;
        X(3) = (i+2) * dx;

        Vector4 Y;
        Y(0) = (j-1) * dx;
        Y(1) = j * dx;
        Y(2) = (j+1) * dx;
        Y(3) = (j+2) * dx;

        Matrix4 Q = q.submat(i-1, j-1, i+2, j+2);

        retval = bicerp(x, y, X, Y, Q, clamp);
    } else { // around boundaries switch to linear interpolation (not enough points to use cubic interpolation)
        retval = grid_bilerp(x, y, q, dx);
    }

    return retval;
}

scalar_t bilerp(scalar_t x, scalar_t y, scalar_t x1, scalar_t y1, scalar_t x2, scalar_t y2, const Matrix2 &Q){
    scalar_t L1 = lerp(x, x1, x2, Q(0,0), Q(1,0));
    scalar_t L2 = lerp(x, x1, x2, Q(0,1), Q(1,1));
    scalar_t r = lerp(y, y1, y2, L1, L2);

    return r;
}

scalar_t lerp(scalar_t x, scalar_t x1, scalar_t x2, scalar_t Q1, scalar_t Q2){
    scalar_t t = (x - x1) / (x2 - x1);

    return Q1*(1.0-t) + t*Q2;
}

scalar_t cerp(scalar_t x, const Vector4 &X, const Vector4 &Q, bool clamp){
    scalar_t val = 0;
    scalar_t s = (x - X(1))/(X(2) - X(1));

    val += ((-1.0/3.0)*s + 0.5*s*s - (1.0/6.0)*s*s*s)*Q(0);
    val += (1.0 - s*s + 0.5*(s*s*s - s))*Q(1);
    val += (s + 0.5*(s*s - s*s*s))*Q(2);
    val += ((1.0/6.0)*(s*s*s - s))*Q(3);

    if(clamp){
        if (val > Q.max()){
            val = Q.max();
        } else if (val < Q.min()){
            val = Q.min();
        }
    }

    return val;
}

scalar_t bicerp(scalar_t x, scalar_t y, Vector4 X, Vector4 Y, Matrix4 &Q, bool clamp){
    Vector4 Qy;
    for (int i = 0; i < 4; i++){
        Vector4 Qx = {Q(0, i), Q(1, i), Q(2, i), Q(3, i)};
        Qy(i) = cerp(x, X, Qx, clamp);
    }

    return cerp(y, Y, Qy, clamp);
}

scalar_t clamp(scalar_t a, scalar_t min, scalar_t max){
    if (a >= max){
        return max;
    } else if (a <= min){
        return min;
    } else {
        return a;
    }
}

scalar_t clamp4(scalar_t a, Matrix2 vals){
    if (a > vals.max()){
        a = vals.max();
    } else if (a < vals.min()){
        a = vals.min();
    }

    return a;
}

scalar_t biclamp(scalar_t val, scalar_t max){
    scalar_t sign = sgn(val);
    scalar_t result = sign*std::min(std::abs(val), std::abs(max));
    return result;
}