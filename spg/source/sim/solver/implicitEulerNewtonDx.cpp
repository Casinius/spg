#include <spg/sim/solver/implicitEulerNewtonDx.h>
#include <spg/sim/simObject/particleGroup.h>
#include <spg/sim/simObject/rigidBodyGroup.h>
#include <spg/utils/timer.h>
#include <spg/utils/functionalUtilities.h>

#include <iostream>
#include "spg/types.h"

namespace spg::solver
{
template <typename MatvecFunc>
VectorX solveLinearSystemPCG(const VectorX &b, MatvecFunc &&matvec, int maxit, double rtol)
{
    const int n = b.size();
    VectorX x = VectorX::Zero(n);
    VectorX r = b;  // r = b - A·x,  x=0
    VectorX p = r;
    double rsold = r.squaredNorm();
    const double b2 = b.squaredNorm();
    if (b2 == 0.0)
        return x;
    const double tol2 = (rtol * rtol) * b2;

    for (int k = 0; k < maxit; ++k) {
        VectorX Ap = matvec(p);  // A·p，走你的回调
        const double alpha = rsold / p.dot(Ap);
        x.noalias() += alpha * p;
        r.noalias() -= alpha * Ap;

        const double rsnew = r.squaredNorm();
        if (rsnew < tol2)
            break;

        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    }
    return x;
}
void ImplicitEulerNewtonDx::step()
{
    if (m_verbosity == Verbosity::Performance) {
        std::cout << "NewtonDx step\n";
    }
    Timer timer;
    timer.start();

    const Real dt = m_dtStep / m_nsubsteps;
    // Compute total DOFs
    int accumulatedNDOF = 0;
    apply_each(
        [&accumulatedNDOF](const auto &objs) {
            for (const auto &obj : objs) {
                accumulatedNDOF += obj.nDOF();
            }
        },
        m_objects);
    const int totalNDOF{accumulatedNDOF};

    std::cout << "[dbg] totalNDOF = " << totalNDOF << "  1×VectorX ≈ " << (totalNDOF * 8LL / 1024 / 1024) << " MB\n";
    std::cout << "[dbg] 预计同时存活 ≈ " << (totalNDOF * 8LL * 7 / 1024 / 1024)
              << " MB (x0,M,f,PCG 4 向量,matvec 2 向量)\n";
    for (int s = 0; s < m_nsubsteps; ++s) {
        // Store state backup
        VectorX x0(totalNDOF);

        getSystemPositions(x0);

        // Compute mass matrix in initial state to prevent simulations with rigid bodies to explode
        SparseMatrix M(totalNDOF, totalNDOF);
        getSystemMassMatrix(M);

        // Set initial guess as inertial position
        integrateObjectsVelocities(dt);

        // // Compute forces and stiffness matrix
        // VectorX f(totalNDOF);
        // SparseMatrix K(totalNDOF, totalNDOF);
        // getSystemForce(f);
        // getSystemStiffnessMatrix(K);
        const auto eps = 1e-4;
        // Create Linear problem left and right hand sides
        const Real invdt = 1. / dt;

        // try {
        for (int iter = 0; iter < 1; ++iter) {
            std::cout << "[dbg] alloc f\n";
            VectorX f(totalNDOF);
            std::cout << "[dbg] getSystemForce(f) begin\n";
            getSystemForce(f);
            std::cout << "[dbg] getSystemForce(f) done ||f||=" << f.norm() << "\n";

            if (f.norm() < m_cgThreshold) {
                std::cout << "[dbg] converged\n";
                break;
            }

            std::cout << "[dbg] enter PCG\n";
            VectorX dx = solveLinearSystemPCG(
                f,
                [&](const VectorX &v) -> VectorX {
                    VectorX x_save(totalNDOF);
                    getSystemPositions(x_save);
                    setSystemPositions(x_save + eps * v);
                    VectorX f_pert(totalNDOF);
                    getSystemForce(f_pert);
                    setSystemPositions(x_save);

                    VectorX Kv = (f_pert - f) / eps;
                    VectorX result = (invdt * invdt) * (M * v) - Kv;  // 显式求值
                    return result;                                    // 返回真正的 VectorX
                },
                200,
                1e-8);
            std::cout << "[dbg] PCG done ||dx||=" << dx.norm() << "\n";

            integrateObjectsVelocitiesFromDx(dx, x0, invdt);
            std::cout << "[dbg] integrated\n";
        }
        // } catch (const std::exception &e) {
        //     std::cerr << "Exception: " << e.what() << "\n";
        //     throw;
        // }
        // // const SparseMatrix LHS = (invdt * invdt) * M - K;
        // const VectorX RHS = f;

        // // Solve problem to obtain dx
        // VectorX dx;
        // solveLinearSystem(LHS, RHS, dx);

        // // Update objects state
        // integrateObjectsVelocitiesFromDx(dx, x0, invdt);
    }
    timer.stop();
    if (m_verbosity == Verbosity::Performance) {
        std::cout << "  Total step time: " << timer.getMilliseconds() << "ms\n\n";
    }
}
}  // namespace spg::solver