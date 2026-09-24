#include <spg/sim/solver/implicitEulerNewtonDx.h>
#include <spg/sim/simObject/particleGroup.h>
#include <spg/sim/simObject/rigidBodyGroup.h>
#include <spg/utils/timer.h>
#include <spg/utils/functionalUtilities.h>

#include <iostream>

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
    const Real invdt = 1.0 / dt;

    // ---------- 总自由度 ----------
    int accumulatedNDOF = 0;
    apply_each(
        [&accumulatedNDOF](const auto &objs) {
            for (const auto &obj : objs) {
                accumulatedNDOF += obj.nDOF();
            }
        },
        m_objects);
    const int totalNDOF{accumulatedNDOF};

    // ---------- 牛顿迭代参数 ----------
    const int maxNewtonIter = 10;
    const Real newtonRtol = 1e-8;

    for (int s = 0; s < m_nsubsteps; ++s) {
        // ================================================================
        // 1. 保存上一步位置 x0
        // ================================================================
        VectorX x0(totalNDOF);
        getSystemPositions(x0);

        // ================================================================
        // 2. 质量矩阵（在上一步位置处，逐子步只算一次）
        // ================================================================
        SparseMatrix M(totalNDOF, totalNDOF);
        getSystemMassMatrix(M);

        // ================================================================
        // 3. 惯性外推: x_tilde = x0 + dt * v0
        //    integrateObjectsVelocities 会原地把每个 obj 的位置推到惯性位置
        // ================================================================
        integrateObjectsVelocities(dt);
        VectorX x_tilde(totalNDOF);
        getSystemPositions(x_tilde);

        // ================================================================
        // 4. 牛顿迭代
        //
        //    残差:      r(x) = M/dt² (x - x_tilde) - f_int(x)
        //    牛顿方程:  (M/dt² - K) δx = -r(x)
        //    同乘 dt²:  (M - dt²·K) δx = -dt²·r(x)
        //
        //    注意: getSystemStiffnessMatrix 内部存的是 negativeHessian,
        //    即 K = ∂f_int/∂x, 所以 (M/dt² - K) 是正确的 Jacobian.
        //    dt² 缩放让条件数从 O(1/dt²) 降到 O(1).
        // ================================================================
        Real r0_norm = 0.0;
        bool newtonConverged = false;

        // 4d. 当前 x 处的刚度矩阵 K = ∂f_int/∂x
        SparseMatrix K(totalNDOF, totalNDOF);
        getSystemStiffnessMatrix(K);
        for (int iter = 0; iter < maxNewtonIter; ++iter) {
            // 4a. 当前内力 f_int(x)
            VectorX f_int(totalNDOF);
            getSystemForce(f_int);

            // 4b. 当前 x
            VectorX x(totalNDOF);
            getSystemPositions(x);

            // 4c. 残差
            VectorX r = (invdt * invdt) * (M * (x - x_tilde)) - f_int;
            const Real rnorm = r.norm();
            if (iter == 0)
                r0_norm = rnorm;

            if (m_verbosity == Verbosity::Performance) {
                std::cout << "  Newton iter " << iter << "  ||r|| = " << rnorm;
                if (iter == 0)
                    std::cout << "  (initial)";
                std::cout << "\n";
            }

            // 收敛判断: 相对容差 + 绝对兜底
            const Real tol = std::max(newtonRtol * r0_norm, Real(1e-14));
            if (rnorm < tol) {
                newtonConverged = true;
                break;
            }

            // 4e. dt² 缩放后的线性系统
            //     LHS = M - dt²·K
            //     RHS = -dt²·r
            const SparseMatrix LHS = M - (dt * dt) * K;
            const VectorX RHS = -(dt * dt) * r;

            // 4f. 求解 δx
            VectorX dx;
            solveLinearSystemLLT(LHS, RHS, dx);

            // 4g. 更新位置
            //     updateObjectsPositionsFromDx 内部对刚体走指数映射,
            //     天然保持单位四元数, 不会引入 NaN.
            updateObjectsPositionsFromDx(dx);
        }

        if (!newtonConverged && m_verbosity == Verbosity::Performance) {
            std::cout << "  Newton did not converge in " << maxNewtonIter << " iters (initial ||r|| = " << r0_norm
                      << ")\n";
        }

        // ================================================================
        // 5. 速度更新
        //
        //    迭代后 obj 内部位置已经是最终 x. 用当前位置与 x0 计算:
        //        v_new = (x - x0) / dt
        //    对刚体, computeIntegratedVelocities 内部会通过四元数对数
        //    映射正确提取角速度.
        // ================================================================
        int offset = 0;
        apply_each(
            [&offset, &x0, invdt](auto &objs) {
                for (auto &obj : objs) {
                    obj.computeIntegratedVelocities(x0, offset, invdt);
                    offset += obj.nDOF();
                }
            },
            m_objects);
    }

    timer.stop();
    if (m_verbosity == Verbosity::Performance) {
        std::cout << "  Total step time: " << timer.getMilliseconds() << "ms\n\n";
    }
}
}  // namespace spg::solver