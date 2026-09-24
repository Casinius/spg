// precond_krylov.h — header-only 预条件 Krylov 框架（PCG / BiCGSTAB / GMRES(m)）
//
// 向量类型固定为 Eigen 动态列向量（与 ODE 求解器的 State<Scalar> 同型，但不依赖 solve.h）。
// 三个求解器统一签名：x 带初值进出；M 为空 std::function = Identity（未预条件）。
// 预条件子 M 表示 M^{-1} 的作用（即 apply 后逼近 A^{-1}）。
//
// 收敛判据（统一）：||r|| <= max(rtol * ||b||, atol)，r 为迭代递推残差；
// 返回前用真残差 ||b - A x|| / ||b|| 复核并填充 Result::relres。
#ifndef PRECOND_KRYLOV_H
#define PRECOND_KRYLOV_H

#include <cmath>
#include <limits>
#include <functional>
#include <iostream>

#include <Eigen/Dense>

namespace pk {

template <class Scalar> using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
template <class Scalar>
using Operator = std::function<Vector<Scalar>(const Vector<Scalar> &)>;

template <class Scalar> struct Params {
  int max_iter = 200;
  Scalar rtol = Scalar(1e-8); // ||r|| <= max(rtol*||b||, atol)
  Scalar atol = Scalar(0);
  int restart = 30;     // 仅 GMRES 使用（每轮 Arnoldi 步数）
  bool verbose = false; // 每步残差打印到 std::cerr
};

template <class Scalar> struct Result {
  bool converged = false;
  int iterations = 0;
  Scalar relres = Scalar(0); // 返回前用真残差 ||b-Ax||/||b|| 复核并填充
};

namespace detail {

template <class Scalar>
inline Vector<Scalar> apply(const Operator<Scalar> &M, Vector<Scalar> v) {
  return M ? M(v) : v;
}

// 统一收尾：真残差复核 + relres 填充。返回 (converged, true_relres)。
template <class Scalar>
inline std::pair<bool, Scalar> finalize(const Operator<Scalar> &A,
                                        const Vector<Scalar> &b,
                                        const Vector<Scalar> &x,
                                        Scalar last_iter_res) {
  const Scalar bnorm = b.norm();
  if (!(bnorm > Scalar(0)))
    return {true, Scalar(0)};
  Vector<Scalar> rtrue = b - A(x);
  if (rtrue.array().isNaN().any())
    return {false, std::numeric_limits<Scalar>::infinity()};
  const Scalar trn = rtrue.norm();
  const Scalar rel = trn / bnorm;
  // 真残差与迭代残差偏差 >10 倍：SPD 破坏/数值漂移信号
  if (last_iter_res > Scalar(0) && trn > Scalar(10) * last_iter_res) {
    std::cerr << "pk: true residual ||b-Ax||=" << trn
              << " deviates >10x from iterated residual " << last_iter_res
              << " (operator may not be SPD / numerically drifting)"
              << std::endl;
  }
  return {trn <= std::numeric_limits<Scalar>::epsilon() * Scalar(100) * bnorm ||
              trn <= Scalar(10) * last_iter_res || last_iter_res <= Scalar(0),
          rel};
}

} // namespace detail

// ----------------------------------------------------------------------
// PCG — 预条件共轭梯度（要求 A 对称正定）
// ----------------------------------------------------------------------
template <class Scalar>
Result<Scalar> pcg(const Operator<Scalar> &A, const Vector<Scalar> &b,
                   Vector<Scalar> &x, const Operator<Scalar> &M = nullptr,
                   const Params<Scalar> &p = {}) {
  Result<Scalar> out;
  if (b.size() == 0) {
    out.converged = true;
    return out;
  }
  const Scalar bnorm = b.norm();
  if (bnorm == Scalar(0)) {
    x.setZero(b.size());
    out.converged = true;
    return out;
  }
  const Scalar tol = std::max(p.rtol * bnorm, p.atol);

  Vector<Scalar> r = b - A(x);
  if (r.array().isNaN().any())
    return out;
  Vector<Scalar> z = detail::apply(M, r);
  Vector<Scalar> dir = z;
  Scalar rho = r.dot(z);
  Scalar rho_prev = Scalar(0);
  Scalar last_res = r.norm();

  for (int it = 0; it < p.max_iter; ++it) {
    if (!(rho > Scalar(0))) // breakdown：rho<=0 或 NaN → 非 SPD/奇异
      break;
    if (it > 0)
      dir = z + (rho / rho_prev) * dir;
    Vector<Scalar> Adir = A(dir);
    Scalar dirAd = dir.dot(Adir);
    if (!(dirAd > Scalar(0))) // p·Ap <= 0 → 非 SPD 或奇异
      break;
    const Scalar alpha = rho / dirAd;
    x += alpha * dir;
    r -= alpha * Adir;
    ++out.iterations;
    if (r.array().isNaN().any())
      break;
    last_res = r.norm();
    if (p.verbose)
      std::cerr << "pk::pcg iter " << out.iterations << " |r|=" << last_res
                << std::endl;
    if (last_res <= tol) {
      out.converged = true;
      break;
    }
    z = detail::apply(M, r);
    rho_prev = rho;
    rho = r.dot(z);
  }
  auto fin = detail::finalize(A, b, x, out.converged ? last_res : Scalar(0));
  out.relres = fin.second;
  return out;
}

// ----------------------------------------------------------------------
// BiCGSTAB — 左预条件（M 只作用于搜索方向），适用于非对称系统
// ----------------------------------------------------------------------
template <class Scalar>
Result<Scalar> bicgstab(const Operator<Scalar> &A, const Vector<Scalar> &b,
                        Vector<Scalar> &x, const Operator<Scalar> &M = nullptr,
                        const Params<Scalar> &p = {}) {
  Result<Scalar> out;
  if (b.size() == 0) {
    out.converged = true;
    return out;
  }
  const Scalar bnorm = b.norm();
  if (bnorm == Scalar(0)) {
    x.setZero(b.size());
    out.converged = true;
    return out;
  }
  const Scalar tol = std::max(p.rtol * bnorm, p.atol);

  Vector<Scalar> r = b - A(x);
  if (r.array().isNaN().any())
    return out;
  const Vector<Scalar> rhat = r; // 残差影子向量
  Scalar rho_old = Scalar(0), alpha = Scalar(0), omega = Scalar(0);
  Vector<Scalar> v = Vector<Scalar>::Zero(b.size());
  Vector<Scalar> pvec = r;
  Scalar last_res = r.norm();

  for (int it = 0; it < p.max_iter; ++it) {
    const Scalar rho = rhat.dot(r);
    if (rho == Scalar(0) || std::isnan(rho))
      break; // breakdown
    if (it > 0) {
      if (omega == Scalar(0))
        break; // breakdown
      const Scalar beta = (rho / rho_old) * (alpha / omega);
      pvec = r + beta * (pvec - omega * v);
    }
    Vector<Scalar> phat = detail::apply(M, pvec);
    v = A(phat);
    const Scalar rhatv = rhat.dot(v);
    if (rhatv == Scalar(0) || std::isnan(rhatv))
      break;
    alpha = rho / rhatv;
    Vector<Scalar> s = r - alpha * v;
    ++out.iterations;
    if (s.array().isNaN().any())
      break;
    last_res = s.norm();
    if (p.verbose)
      std::cerr << "pk::bicgstab iter " << out.iterations << " |s|=" << last_res
                << std::endl;
    if (last_res <= tol) {
      x += alpha * phat;
      out.converged = true;
      break;
    }
    Vector<Scalar> shat = detail::apply(M, s);
    Vector<Scalar> t = A(shat);
    const Scalar tt = t.dot(t);
    if (tt == Scalar(0) || std::isnan(tt))
      break;
    omega = t.dot(s) / tt;
    if (omega == Scalar(0) || std::isnan(omega))
      break; // breakdown
    x += alpha * phat + omega * shat;
    r = s - omega * t;
    rho_old = rho;
    if (r.array().isNaN().any())
      break;
    last_res = r.norm();
    if (p.verbose)
      std::cerr << "pk::bicgstab iter " << out.iterations << " |r|=" << last_res
                << std::endl;
    if (last_res <= tol) {
      out.converged = true;
      break;
    }
  }
  auto fin = detail::finalize(A, b, x, out.converged ? last_res : Scalar(0));
  out.relres = fin.second;
  return out;
}

// ----------------------------------------------------------------------
// GMRES(m) — 重启式，左预条件，修正 Gram-Schmidt + Givens 旋转
// ----------------------------------------------------------------------
template <class Scalar>
Result<Scalar> gmres(const Operator<Scalar> &A, const Vector<Scalar> &b,
                     Vector<Scalar> &x, const Operator<Scalar> &M = nullptr,
                     const Params<Scalar> &p = {}) {
  Result<Scalar> out;
  if (b.size() == 0) {
    out.converged = true;
    return out;
  }
  const Scalar bnorm = b.norm();
  if (bnorm == Scalar(0)) {
    x.setZero(b.size());
    out.converged = true;
    return out;
  }
  const Scalar tol = std::max(p.rtol * bnorm, p.atol);
  const int m = p.restart > 0 ? p.restart : 30;
  const int max_outer = std::max(1, p.max_iter / m);
  const int n = b.size();
  Scalar last_res = Scalar(-1);

  for (int outer = 0; outer < max_outer; ++outer) {
    Vector<Scalar> resid = b - A(x);
    Vector<Scalar> r = detail::apply(M, resid); // 左预条件残差
    if (r.array().isNaN().any())
      return out;
    const Scalar beta = r.norm();
    if (p.verbose)
      std::cerr << "pk::gmres outer " << outer << " |r|=" << beta << std::endl;
    if (beta <= tol) {
      out.converged = true;
      break;
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> V(n, m + 1);
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> H =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(m + 1, m);
    Vector<Scalar> g = Vector<Scalar>::Zero(m + 1);
    Vector<Scalar> cs = Vector<Scalar>::Zero(m);
    Vector<Scalar> sn = Vector<Scalar>::Zero(m);
    V.col(0) = r / beta;
    g(0) = beta;

    int j = 0;
    bool happy = false;
    for (j = 0; j < m; ++j) {
      Vector<Scalar> w = detail::apply(M, A(V.col(j)));
      if (w.array().isNaN().any())
        return out;
      // 修正 Gram-Schmidt（两次通过，第二次消除舍入正交性损失）
      for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i <= j; ++i) {
          const Scalar c = V.col(i).dot(w);
          w -= c * V.col(i);
          if (pass == 0)
            H(i, j) = c;
          else
            H(i, j) += c;
        }
      }
      const Scalar hnext = w.norm();
      H(j + 1, j) = hnext;
      ++out.iterations;
      if (hnext == Scalar(0)) {
        happy = true; // Krylov 空间已张成，精确解在内
        break;
      }
      V.col(j + 1) = w / hnext;

      // 已有 Givens 旋转作用到新列
      for (int i = 0; i < j; ++i) {
        const Scalar t1 = cs(i) * H(i, j) + sn(i) * H(i + 1, j);
        const Scalar t2 = -sn(i) * H(i, j) + cs(i) * H(i + 1, j);
        H(i, j) = t1;
        H(i + 1, j) = t2;
      }
      // 新旋转（消去 H(j+1,j)）
      const Scalar denom = std::hypot(H(j, j), H(j + 1, j));
      if (denom == Scalar(0)) {
        cs(j) = Scalar(1);
        sn(j) = Scalar(0);
      } else {
        cs(j) = H(j, j) / denom;
        sn(j) = H(j + 1, j) / denom;
      }
      H(j, j) = denom;
      H(j + 1, j) = Scalar(0);
      const Scalar gtmp = cs(j) * g(j) + sn(j) * g(j + 1);
      g(j + 1) = -sn(j) * g(j) + cs(j) * g(j + 1);
      g(j) = gtmp;
      last_res = std::abs(g(j + 1));
      if (p.verbose)
        std::cerr << "pk::gmres iter " << out.iterations
                  << " |res|=" << last_res << std::endl;
      if (last_res <= tol)
        break;
    }

    // 解上三角最小二乘 H(0:j,0:j) y = g(0:j)
    const int k = (j < m) ? j + 1 : m;
    Vector<Scalar> y = g.head(k);
    auto Htop = H.topLeftCorner(k, k);
    Htop.template triangularView<Eigen::Upper>().solveInPlace(y);
    x += V.leftCols(k) * y;

    if (happy)
      continue; // 解已更新，下一外轮残差检查会确认收敛
    if (last_res >= Scalar(0) && last_res <= tol) {
      out.converged = true;
      break;
    }
  }
  auto fin = detail::finalize(A, b, x, out.converged ? last_res : Scalar(-1));
  out.relres = fin.second;
  return out;
}

// ----------------------------------------------------------------------
// 预条件子库（全部返回 M^{-1} 作用的 Operator）
// ----------------------------------------------------------------------

template <class Scalar> Operator<Scalar> identity_precond() {
  return [](const Vector<Scalar> &v) { return v; };
}

// Jacobi：M^{-1} = D^{-1}。零对角元按 1 处理（不除零）。
template <class Scalar>
Operator<Scalar> jacobi(const Vector<Scalar> &diag) {
  return [diag](const Vector<Scalar> &v) {
    Vector<Scalar> out(v.size());
    for (int i = 0; i < v.size(); ++i)
      out(i) = diag(i) != Scalar(0) ? v(i) / diag(i) : v(i);
    return out;
  };
}

template <class Scalar>
Operator<Scalar>
jacobi(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A) {
  return jacobi<Scalar>(Vector<Scalar>(A.diagonal()));
}

// SSOR：M = (D+ωL) D^{-1} (D+ωU)，M^{-1} v = ω(2-ω)·后继回代(D·前代(v))。
// 显式临时向量实现，避免任何 in-place 别名问题。
template <class Scalar>
Operator<Scalar>
ssor(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A,
     Scalar omega = Scalar(1)) {
  const int n = A.rows();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> L =
      A.template triangularView<Eigen::StrictlyLower>().toDenseMatrix();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> U =
      A.template triangularView<Eigen::StrictlyUpper>().toDenseMatrix();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> D =
      A.diagonal().asDiagonal();
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DL = D + omega * L;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DU = D + omega * U;
  return [DL, DU, D, omega](const Vector<Scalar> &v) -> Vector<Scalar> {
    Vector<Scalar> y = v;
    DL.template triangularView<Eigen::Lower>().solveInPlace(y); // 前代
    Vector<Scalar> z = D * y;
    DU.template triangularView<Eigen::Upper>().solveInPlace(z); // 回代
    Vector<Scalar> out = (omega * (Scalar(2) - omega)) * z;
    return out;
  };
}





} // namespace pk



// spg_krylov_bridge.h
#include <spg/sim/simObject/particleGroup.h>
#include <spg/sim/simObject/rigidBodyGroup.h>
#include <spg/utils/timer.h>
#include <spg/utils/functionalUtilities.h>
namespace spg::solver::krylov_bridge {

// SparseMatrix -> pk::Operator（按引用捕获，调用方保证生命周期）
template <class Scalar>
inline pk::Operator<Scalar> makeOperator(const SparseMatrix& A) {
    return [&A](const pk::Vector<Scalar>& v) -> pk::Vector<Scalar> {
        return A * v;          // Eigen 稀疏矩阵向量乘
    };
}

// 从 LHS 对角提取 Jacobi 预条件子（对刚体块对角还可用块版本，见下文）
template <class Scalar>
inline pk::Operator<Scalar> makeJacobi(const SparseMatrix& A) {
    pk::Vector<Scalar> d = A.diagonal();
    // 防零：pk::jacobi 内部已做除零保护，这里只处理显式零元
    for (int i = 0; i < d.size(); ++i)
        if (d(i) == Scalar(0)) d(i) = Scalar(1);
    return pk::jacobi<Scalar>(d);
}

} // namespace spg::solver::krylov_bridge

#endif // PRECOND_KRYLOV_H