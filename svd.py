""" Contains some SVD functions """

__author__ = "Tibor Schneider"
__email__ = "sctibor@student.ethz.ch"

import numpy as np
import unittest

# with 1e-4, we get 1e-1 accuracy, requiring approx 360 iterations
SVD_EPSILON = 1e-4 # 1e-15 # 2e-4
_ACCURACY = 1e-1 # with epsilon:1e-4
MIN_ALLOWED_EIGENVALUE = 1e-3

def logm(mat, epsilon=SVD_EPSILON):
    """ Computes the matrix logarithm of a real, symmetric matrix

    Parameters
    ----------

    mat: np.array, size=(N, N)
         Input matrix, must be real and symmetric

    Returns
    -------
    np.array, size=(N, N): result
    """
    # convert the matrix to float32
    mat = np.float32(mat)
    L, D, R = svd(mat, epsilon=epsilon)
    D = np.clip(D, MIN_ALLOWED_EIGENVALUE, None)
    log_D = np.diag(np.log(np.diag(D)))
    return L @ log_D @ R


def svd(mat, epsilon=SVD_EPSILON, with_n_iter=False):
    """ Computes the singular value decomposition.

    A = L D R, where A is the input, D is a diagonal matrix, L and R are orthogonal matrices

    Parameters
    ----------

    mat: np.array, size=(N, N)
         Input matrix, must be real and symmetric

    Returns
    -------

    np.array, size=(N, N): L, orthogonal matrix
    np.array, size=(N, N): D, diagonal matrix, containing the eigenvalues of A
    np.array, size=(N, N): R, orthogonal matrix
    """
    # return _jacobi_eigv(mat)
    Lt, T, Rt = _householder_tridiagonal(mat)
    eigvals, Q, n_iter = _qr_symm_tridiag(T, epsilon=epsilon, with_n_iter=True)
    D = np.diag(eigvals)
    L = Lt @ Q
    R = Q.T @ Rt
    if with_n_iter:
        return L, D, R, n_iter
    return L, D, R


QR_MAX_N_REPEAT = 100


def _qr_symm_tridiag(T, epsilon=SVD_EPSILON, with_n_iter=False):
    """ Computes the Eigenvalues and Eigenvectors of a real, symmetric, tridiagonal matrix.

    The eigenvalues are computed via implicit wilkinson shift

    See people.inf.ethz.ch/arbenz/wep/Lnotes/chapter4.pdf

    Parameters
    ----------

    mat: np.array, size=(N, N)
         Input matrix, must be real, symmetric and tridiagonal

    Returns
    -------

    np.array, size=(N, ): eigenvalues
    np.array, size=(N, N): eigenvectors
    """

    main_diag = np.diag(T).copy()
    off_diag = np.diag(T, k=1).copy() # range 0..N-1, in reference, it is 1..N (also, they use 1-indexing)

    return _qr_symm_tridiag_work(main_diag, off_diag, epsilon=epsilon, with_n_iter=with_n_iter)


def _qr_symm_tridiag_work(main_diag, off_diag, epsilon=SVD_EPSILON, with_n_iter=False):
    """ Work funciton, which might call itself recursively to solve defaltion """
    N = main_diag.shape[0]
    assert len(off_diag) == N - 1
    Q = np.eye(N).astype(np.float32)
    m = N-1
    n_repeat = QR_MAX_N_REPEAT

    # numerical precisisons
    eps = np.finfo(np.float32).eps
    eps_sqr = eps ** np.float32(2)
    abs_min = np.finfo(np.float32).tiny
    safe_min = np.sqrt(abs_min) / eps_sqr
    safe_max = np.sqrt(np.float32(1) / abs_min) / np.float32(3)

    # check recursive break
    if N == 1:
        if with_n_iter:
            return main_diag, np.eye(1).astype(np.float32), 0
        return main_diag, np.eye(1).astype(np.float32)

    if N == 2:
        # diagonalize this matrix quickly
        # the formula for calculating the two eigenvalues, with main_diag[k] = ak and offdiag[0] = b
        # lambda1 = a1*c^2 - 2 b*c*s + a2 * s^2
        # lambda2 = a1*s^2 + 2 b*c*s + a2 * c^2
        c, s = _givens_diag(main_diag[0], off_diag[0], main_diag[1])
        c_square = c * c
        s_square = s * s
        bcs2 = off_diag[0] * c * s * np.float32(2)
        lambda1 = main_diag[0] * c_square - bcs2 + main_diag[1] * s_square
        lambda2 = main_diag[0] * s_square + bcs2 + main_diag[1] * c_square
        if with_n_iter:
            return np.array([lambda1, lambda2]), np.array([[c, -s], [s, c]]), 1
        return np.array([lambda1, lambda2]), np.array([[c, -s], [s, c]])

    if with_n_iter:
        n_iter = 0

    while m > 0:

        # check if the matrix which is left to be transformed has zero offdiagonal elements
        for k in range(m-1):
            if np.abs(off_diag[k]) < eps:
                # the off-diag element k is very small! divide the matrix
                main_diag[:k+1], Q1, n_iter1 = _qr_symm_tridiag_work(main_diag[:k+1], off_diag[:k], epsilon=epsilon, with_n_iter=True)
                main_diag[k+1:m+1], Q2, n_iter2 = _qr_symm_tridiag_work(main_diag[k+1:m+1], off_diag[k+1:m], epsilon=epsilon, with_n_iter=True)
                # reproduce the Q
                Q_construct = np.eye(N).astype(np.float32)
                Q_construct[:k+1, :k+1] = Q1
                Q_construct[k+1:m+1, k+1:m+1] = Q2
                Q = Q @ Q_construct
                if with_n_iter:
                    return main_diag, Q, n_iter + n_iter1 + n_iter2
                return main_diag, Q

        # do wilkinson shift
        d = (main_diag[m-1] - main_diag[m]) / np.float32(2)
        if d == 0:
            shift = main_diag[m] - np.abs(off_diag[m-1])
        else:
            shift = main_diag[m] - ((off_diag[m-1] ** np.float32(2)) / (d + np.sign(d) * np.sqrt(d ** np.float32(2) + off_diag[m-1] ** np.float32(2))))

        # start the implicit QR step
        x = main_diag[0] - shift
        y = off_diag[0]
        for k in range(m): # k will at most be m-1
            # determine the givens rotation
            if m > 1:
                c, s = _givens(x, y)
            else:
                # diagonalize the remaining elements, only done once
                c, s = _givens_diag(main_diag[0], off_diag[0], main_diag[1])

            # compute some values
            w = c * x - s * y
            d = main_diag[k] - main_diag[k + 1]
            z = (np.float32(2) * c * off_diag[k] + d * s) * s

            # do the step on the main and off diagonal
            main_diag[k] = main_diag[k] - z
            main_diag[k + 1] = main_diag[k + 1] + z
            off_diag[k] = d * c * s + (c * c - s * s) * off_diag[k]
            if k > 0:
                off_diag[k - 1] = w

            # update x and y
            x = off_diag[k]
            if k < m - 1:
                y = -s * off_diag[k + 1]
                off_diag[k + 1] = c * off_diag[k + 1]

            # update the eigenvectors
            Q[:, k:k + 2] = Q[:, k:k + 2] @ np.array([[c, s], [-s, c]])

            if with_n_iter:
                n_iter += 1

        # check for convergence
        if np.abs(off_diag[m - 1]) < epsilon * (np.abs(main_diag[m - 1]) + np.abs(main_diag[m])):
            m = m - 1
            n_repeat = QR_MAX_N_REPEAT
        else:
            n_repeat -= 1
            if n_repeat == 0:
                np.set_printoptions(linewidth=200)
                print(f"{m=}")
                print(f"{main_diag=}")
                print(f"{off_diag=}")
                raise RuntimeError("Maximum Iterations Reached")

    if with_n_iter:
        return main_diag, Q, n_iter
    return main_diag, Q



def _householder_tridiagonal(mat):
    """ Transforms the matrix into tridiagonal form.

    A Matrix is tridiagonal if all elements not on the main diagonal or one above and belos are zero.

    see www.mcs.csueastbay.edu/~malek/Class/Householder.pdf

    A = L^T T R^T

    Parameters
    ----------

    mat: np.array, size=(N, N)
         Input matrix, must be real and symmetric

    Returns
    -------

    np.array, size=(N, N): L: orthogonal matrix
    np.array, size=(N, N): T: real, symmetric, tridiagonal matrix, s.t. A = L T R
    np.array, size=(N, N): R: orthogonal matrix
    """

    assert mat.shape[1] == mat.shape[0]
    N = mat.shape[0]
    T = mat.copy()
    L = np.eye(N).astype(np.float32)
    R = np.eye(N).astype(np.float32)
    for k in range(N-2):
        s = np.linalg.norm(T[k+1:, k])
        if s == 0:
            continue
        val = T[k+1, k]
        sign = np.sign(val)
        z = (np.float32(1) + sign * val / s) / np.float32(2)
        sqrtz = np.sqrt(z)
        v = np.zeros(N).astype(np.float32)
        v[k+1] = sqrtz
        v[k+2:] = (sign * T[k, k+2:]) / (np.float32(2) * s * sqrtz)
        v = v.reshape(-1, 1)
        H = np.eye(N).astype(np.float32) - np.float32(2) * v @ v.T
        T = H @ T @ H
        L = H @ L
        R = R @ H

    return L.T, T, R.T


GIVENS_SAVE_MINIMUM = 1e-10


def _givens(a, b):
    """ Computes the parameters for the givens rotation.

    The values c = cos(theta), s = sin(theta) are computed, such that:

    | c -s | | a | = | r |
    | s  c | | b |   | 0 |

    Parameters
    ----------

    a: float
    b: float

    Returns
    -------
    float: c = cos(theta)
    float: s = sin(theta)
    """
    if b == 0:
        c = np.float32(1)
        s = np.float32(0)
    elif a == 0:
        c = np.float32(0)
        s = np.sign(b)
    else:
        # scale a and b to avoid underflow / overflow
        scale = max(abs(a), abs(b))
        if scale < GIVENS_SAVE_MINIMUM:
            a = a / scale
            b = b / scale
        r = np.sqrt(a ** np.float32(2) + b ** np.float32(2))
        inv_r = np.float32(1)/r
        c = a * inv_r
        s = -b * inv_r
    return c, s


def _givens_diag(a, b, c):
    """ Computes the parameters for the givens rotation.

    The values c = cos(theta), s = sin(theta) are computed, such that:

    | c -s |^T | a  b | | c -s |   | p  0 |
    | s  c |   | b  c | | s  c | = | 0  q |

    Parameters
    ----------

    a: float
    b: float: off-diagonal
    c: float

    Returns
    -------
    float: c = cos(theta)
    float: s = sin(theta)
    """
    if b == 0:
        return 1, 0
    double_angle_tan = (np.float32(2) * b) / (a - c)
    angle = np.arctan(double_angle_tan) / np.float32(2)
    return np.cos(angle), -np.sin(angle)


def _compare_mat(A, B, epsilon=1e-7):
    """ Compares A and B, and returns True if they are very similar """
    return np.all((A - B) < epsilon)


def _is_diag(A, epsilon=1e-7):
    """ returns True if the matrix is diagonal """
    return np.all((A - np.diag(np.diag(A))) < epsilon)


def _is_tridiagonal(A, epsilon=1e-7):
    """ returns true if the matrix is tridiagonal """
    is_ok = True
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            if np.abs(i - k) > 1:
                if A[i, k] > epsilon:
                    is_ok = False
    return is_ok


class TestSVD(unittest.TestCase):
    """ Test SVD functions """
    def test_givens(self):
        """ test _givens function """
        eps = 1e-7
        for _ in range(10):
            a = np.random.randn(1)[0] * 5
            b = np.random.randn(1)[0] * 5
            c, s = _givens(a, b)
            rot = np.array([[c, -s], [s, c]])
            assert _compare_mat(rot @ rot.T, np.eye(2), eps)
            assert (rot @ np.array([[a], [b]]))[1] < eps

    def test_divens_diag(self):
        """ test _givens_diag function """
        for _ in range(100):
            a1 = np.random.randn(1)[0] * 5
            a2 = np.random.randn(1)[0] * 5
            b = np.random.randn(1)[0] * 4
            c, s = _givens_diag(a1, b, a2)
            rot = np.array([[c, -s], [s, c]])
            A = np.array([[a1, b], [b, a2]])
            assert _is_diag(rot @ A @ rot.T)

    def test_householder_tridiagonal(self):
        """ test _householder_tridiagonal """
        for _ in range(10):
            X = np.random.randn(22, 825)
            A = X @ X.T
            L, T, R = _householder_tridiagonal(A)
            assert _is_tridiagonal(T)
            assert _compare_mat(A, L @ T @ R)

    def test_qr_symm_tridiag(self):
        """ test _qr_symm_tridiag """
        eps = _ACCURACY
        for _ in range(10):
            X = np.random.randn(22, 825)
            A = X @ X.T
            _, T, _ = _householder_tridiagonal(A)
            acq_eigvals, Q = _qr_symm_tridiag(T)
            D = np.diag(acq_eigvals)
            exp_eigvals = np.sort(np.linalg.eigvals(T))
            acq_eigvals = np.sort(acq_eigvals)
            assert _compare_mat(T, Q @ D @ Q.T, epsilon=eps)
            assert _compare_mat(exp_eigvals, acq_eigvals, epsilon=eps)

    def test_svd(self):
        """ test svd """
        eps = _ACCURACY
        for _ in range(10):
            X = np.random.randn(22, 825)
            A = X @ X.T
            L, D, R = svd(A)
            exp_eigvals = np.sort(np.linalg.eigvals(A))
            acq_eigvals = np.sort(np.diag(D))
            assert _is_diag(D)
            assert _compare_mat(A, L @ D @ R, epsilon=eps)
            assert _compare_mat(exp_eigvals, acq_eigvals, epsilon=eps)


def _determine_epsilon_for_accuracy(accuracy=1e-4):
    current_eps = accuracy
    success = False
    n_trials = 20
    quantization_levels = 2**16

    while not success:
        success = True
        mean_n_iter = 0
        try:
            for _ in range(n_trials):
                X = np.random.randn(22, 825)
                A = X @ X.T
                max_A = np.abs(A).max()
                A = np.round((A / max_A) * (quantization_levels))
                A = A / quantization_levels * max_A
                A = A.astype(np.float32)
                L, D, R, n_iter = svd(A, epsilon=current_eps, with_n_iter=True)
                mean_n_iter += n_iter / n_trials
                exp_eigvals = np.sort(np.linalg.eigvals(A))
                acq_eigvals = np.sort(np.diag(D))
                assert _compare_mat(A, L @ D @ R, epsilon=accuracy)
                assert _compare_mat(exp_eigvals, acq_eigvals, epsilon=accuracy)
        except AssertionError:
            success = False
            current_eps = current_eps * 0.1
            if current_eps < 1e-30:
                raise RuntimeError("Max number of iterations reached")
    return current_eps, mean_n_iter


def _epsilon_sweep():
    for accuracy in [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        required_eps, n_iter = _determine_epsilon_for_accuracy(accuracy)
        print(f"For accuracy: {accuracy:.0E}, required epsilon: {required_eps:.0E}, n_iter: {n_iter:.1f}")

if __name__ == "__main__":
    _epsilon_sweep()
