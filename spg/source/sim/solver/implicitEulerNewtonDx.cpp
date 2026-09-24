#include <spg/sim/solver/implicitEulerNewtonDx.h>
#include <spg/sim/simObject/particleGroup.h>
#include <spg/sim/simObject/rigidBodyGroup.h>
#include <spg/utils/timer.h>
#include <spg/utils/functionalUtilities.h>

#include <iostream>
#include "pcg.hpp"
#include "spg/types.h"
namespace spg::solver
{
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

    for (int s = 0; s < m_nsubsteps; ++s) {
        // Store state backup
        VectorX x0(totalNDOF);
        getSystemPositions(x0);

        // Compute mass matrix in initial state to prevent simulations with rigid bodies to explode
        SparseMatrix M(totalNDOF, totalNDOF);
        getSystemMassMatrix(M);

        // Set initial guess as inertial position
        integrateObjectsVelocities(dt);

        // Compute forces and stiffness matrix
        VectorX f(totalNDOF);
        SparseMatrix K(totalNDOF, totalNDOF);
        getSystemForce(f);
        getSystemStiffnessMatrix(K);

        // Create Linear problem left and right hand sides
        const Real invdt = 1. / dt;
        const SparseMatrix LHS = (invdt * invdt) * M - K;
        const VectorX RHS = f;

        auto Aop = krylov_bridge::makeOperator<Real>(LHS);
        auto Mop = krylov_bridge::makeJacobi<Real>(LHS);
        pk::Params<Real> p;

        p.rtol = Real(1e-6);
        p.atol = Real(1e-12);

        p.verbose = false;
        VectorX dx = VectorX::Zero(RHS.size());
        pk::Result<Real> r;

        // M - dt^2 K 通常 SPD（dt 足够小），若你担心 K 非对称/大 dt，用 BiCGSTAB
        if (true) {
            r = pk::pcg<Real>(Aop, RHS, dx, Mop, p);
        } else {
            r = pk::bicgstab<Real>(Aop, RHS, dx, Mop, p);
        }

        // 兜底：Krylov 失败/发散 → 退回原直接法，保证帧率不崩
        if (!r.converged) {
            dx.setZero();
            solveLinearSystem(LHS, RHS, dx);
        }
        // Solve problem to obtain dx
        // VectorX dx;
        // solveLinearSystem(LHS, RHS, dx);

        // Update objects state
        integrateObjectsVelocitiesFromDx(dx, x0, invdt);
    }
    timer.stop();
    if (m_verbosity == Verbosity::Performance) {
        std::cout << "  Total step time: " << timer.getMilliseconds() << "ms\n\n";
    }
}
}  // namespace spg::solver