#include <spg/sim/solver/implicitEulerNewtonDv.h>
#include <spg/sim/simObject/particleGroup.h>
#include <spg/sim/simObject/rigidBodyGroup.h>
#include <spg/utils/timer.h>
#include <spg/utils/functionalUtilities.h>

#include <iostream>
#include "pcg.hpp"
#include "spg/types.h"
namespace spg::solver
{
void ImplicitEulerNewtonDv::step()
{
    constexpr int kMaxKiter = 30;

    if (m_verbosity == Verbosity::Performance) {
        std::cout << "NewtonDv step\n";
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
    SparseMatrix K(totalNDOF, totalNDOF);
    SparseMatrix M(totalNDOF, totalNDOF);
    VectorX x0(totalNDOF);
    VectorX v0(totalNDOF);
    VectorX f(totalNDOF);
    

    for (int s = 0; s < m_nsubsteps; ++s) {
        // Store state backup

        getSystemPositions(x0);
        getSystemVelocities(v0);

        // Compute mass matrix in initial state to prevent simulations with rigid bodies to explode

        getSystemMassMatrix(M);

        // Set initial guess as inertial position
        integrateObjectsVelocities(dt);

        // Compute forces and stiffness matrix

        getSystemForce(f);
        getSystemStiffnessMatrix(K);

        // Create Linear problem left and right hand sides
        const SparseMatrix LHS = M - (dt * dt) * K;
        const VectorX RHS = dt * f;

        auto Aop = krylov_bridge::makeOperator<Real>(LHS);
        auto Mop = krylov_bridge::makeJacobi<Real>(LHS);
        pk::Params<Real> p;

        p.rtol = Real(1e-6);
        p.atol = Real(1e-12);

        p.verbose = false;

        // Solve problem to obtain dv
        // VectorX dv;
        // solveLinearSystem(LHS, RHS, dv);
        VectorX dv = VectorX::Zero(RHS.size());
        pk::Result<Real> r;

        // M - dt^2 K 通常 SPD（dt 足够小），若你担心 K 非对称/大 dt，用 BiCGSTAB
        if (true) {
            r = pk::pcg<Real>(Aop, RHS, dv, Mop, p);
        } else {
            r = pk::bicgstab<Real>(Aop, RHS, dv, Mop, p);
        }

        // 兜底：Krylov 失败/发散 → 退回原直接法，保证帧率不崩
        if (!r.converged) {
            dv.setZero();
            solveLinearSystem(LHS, RHS, dv);
        }
        // Update objects state
        setObjectsPositions(x0);
        setObjectsVelocities(v0 + dv);
        integrateObjectsVelocities(dt);
    }
    timer.stop();
    if (m_verbosity == Verbosity::Performance) {
        std::cout << "  Total step time: " << timer.getMilliseconds() << "ms\n\n";
    }
}
}  // namespace spg::solver