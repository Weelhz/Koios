import numpy as np
import scipy.sparse as sp_sparse
import scipy.sparse.linalg as sp_linalg
from scipy.fft import fft, ifft, fft2, ifft2, fftn, ifftn
from scipy.optimize import minimize, differential_evolution
from scipy.special import comb
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import warnings

class SolverType(Enum):
    """Types of numerical solvers"""
    DIRECT = "DIRECT"
    ITERATIVE = "ITERATIVE"
    SPECTRAL = "SPECTRAL"
    MULTIGRID = "MULTIGRID"
    MONTE_CARLO = "MONTE_CARLO"

@dataclass
class NumericalSolution:
    """Container for numerical solution data"""
    solution: np.ndarray
    error_estimate: Optional[float] = None
    iterations: Optional[int] = None
    convergence_history: Optional[List[float]] = None
    method_used: Optional[str] = None
    computation_time: Optional[float] = None

class NumericalMethodsEngine:
    """Advanced numerical methods engine"""
    
    def __init__(self):
        self.tolerance = 1e-10
        self.max_iterations = 1000
    
    def newton_method(self, func_str: str, x0: float, max_iter: int = 100) -> Dict[str, Any]:
        """Newton's method for finding roots of a function"""
        import sympy as sp
        
        x = sp.Symbol('x')
        try:
            # Parse the function
            f = sp.sympify(func_str)
            f_prime = sp.diff(f, x)
            
            # Convert to numerical functions
            f_num = sp.lambdify(x, f)
            f_prime_num = sp.lambdify(x, f_prime)
            
            # Newton's method iteration
            x_current = x0
            iterations = []
            
            for i in range(max_iter):
                try:
                    f_val = f_num(x_current)
                    f_prime_val = f_prime_num(x_current)
                    
                    if abs(f_prime_val) < 1e-10:
                        break
                    
                    x_next = x_current - f_val / f_prime_val
                    iterations.append({
                        'iteration': i,
                        'x': x_current,
                        'f_x': f_val,
                        'f_prime_x': f_prime_val
                    })
                    
                    if abs(x_next - x_current) < self.tolerance:
                        x_current = x_next
                        break
                    
                    x_current = x_next
                except:
                    break
            
            return {
                'success': True,
                'root': x_current,
                'iterations': len(iterations),
                'convergence_history': iterations,
                'final_residual': abs(f_num(x_current))
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        
    def spectral_poisson_solver_2d(self, f: np.ndarray, 
                                  domain: Tuple[float, float, float, float],
                                  boundary_conditions: str = 'dirichlet') -> np.ndarray:
        """
        Solve 2D Poisson equation ∇²u = f using spectral methods
        
        Args:
            f: Right-hand side function values on grid
            domain: (x_min, x_max, y_min, y_max)
            boundary_conditions: 'dirichlet' or 'periodic'
        """
        ny, nx = f.shape
        x_min, x_max, y_min, y_max = domain
        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)
        
        if boundary_conditions == 'periodic':
            # FFT approach for periodic BCs
            f_hat = fft2(f)
            
            # Wave numbers
            kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
            ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
            kx_grid, ky_grid = np.meshgrid(kx, ky)
            
            # Solve in Fourier space: -k² û = f̂
            k_squared = kx_grid**2 + ky_grid**2
            k_squared[0, 0] = 1  # Avoid division by zero
            
            u_hat = -f_hat / k_squared
            u_hat[0, 0] = 0  # Set constant mode to zero
            
            # Transform back
            u = np.real(ifft2(u_hat))
            
        else:  # Dirichlet
            # Use sine transform for homogeneous Dirichlet BCs
            # Extend to sine series
            f_extended = np.zeros((2*ny, 2*nx))
            f_extended[1:ny, 1:nx] = f[1:-1, 1:-1]
            f_extended[ny+1:2*ny, 1:nx] = -f[ny-2:0:-1, 1:-1]
            f_extended[1:ny, nx+1:2*nx] = -f[1:-1, nx-2:0:-1]
            f_extended[ny+1:2*ny, nx+1:2*nx] = f[ny-2:0:-1, nx-2:0:-1]
            
            # FFT
            f_hat = fft2(f_extended)
            
            # Solve
            kx = np.pi * np.arange(2*nx) / (nx * dx)
            ky = np.pi * np.arange(2*ny) / (ny * dy)
            kx_grid, ky_grid = np.meshgrid(kx, ky)
            
            k_squared = kx_grid**2 + ky_grid**2
            k_squared[0, 0] = 1
            
            u_hat = -f_hat / k_squared
            u_hat[0, 0] = 0
            
            # Transform back and extract solution
            u_extended = np.real(ifft2(u_hat))
            u = np.zeros((ny, nx))
            u[1:-1, 1:-1] = u_extended[1:ny, 1:nx]
        
        return u
    
    def multigrid_v_cycle(self, A: sp_sparse.csr_matrix, b: np.ndarray,
                         x0: np.ndarray, levels: int = 3,
                         nu1: int = 2, nu2: int = 2) -> np.ndarray:
        """
        Multigrid V-cycle for solving Ax = b
        
        Args:
            A: System matrix (sparse)
            b: Right-hand side
            x0: Initial guess
            levels: Number of multigrid levels
            nu1: Pre-smoothing iterations
            nu2: Post-smoothing iterations
        """
        n = len(b)
        
        # Create restriction and prolongation operators
        def create_operators(n_fine):
            n_coarse = n_fine // 2
            # Simple injection restriction
            R = sp_sparse.lil_matrix((n_coarse, n_fine))
            for i in range(n_coarse):
                R[i, 2*i] = 0.5
                R[i, 2*i+1] = 0.5
            # Linear interpolation prolongation
            P = 2 * R.T
            return R.tocsr(), P.tocsr()
        
        # Build hierarchy
        A_levels = [A]
        R_levels = []
        P_levels = []
        
        n_current = n
        for level in range(levels - 1):
            if n_current <= 4:
                break
            R, P = create_operators(n_current)
            R_levels.append(R)
            P_levels.append(P)
            A_coarse = R @ A_levels[-1] @ P
            A_levels.append(A_coarse)
            n_current = n_current // 2
        
        def smooth(A, x, b, iterations):
            """Gauss-Seidel smoother"""
            D = A.diagonal()
            for _ in range(iterations):
                for i in range(len(x)):
                    row = A.getrow(i)
                    x[i] = (b[i] - row.dot(x) + D[i] * x[i]) / D[i]
            return x
        
        def v_cycle_recursive(x, b, level):
            """Recursive V-cycle"""
            if level == len(A_levels) - 1:
                # Coarsest level - direct solve
                return sp_linalg.spsolve(A_levels[level], b)
            
            # Pre-smooth
            x = smooth(A_levels[level], x.copy(), b, nu1)
            
            # Compute residual
            r = b - A_levels[level] @ x
            
            # Restrict
            r_coarse = R_levels[level] @ r
            
            # Recursive call
            e_coarse = np.zeros(len(r_coarse))
            e_coarse = v_cycle_recursive(e_coarse, r_coarse, level + 1)
            
            # Prolongate and correct
            x = x + P_levels[level] @ e_coarse
            
            # Post-smooth
            x = smooth(A_levels[level], x, b, nu2)
            
            return x
        
        return v_cycle_recursive(x0, b, 0)
    
    def adaptive_integration(self, f: Callable, a: float, b: float,
                           tol: float = 1e-8, max_depth: int = 50) -> Tuple[float, float]:
        """
        Adaptive quadrature using Simpson's rule with error estimation
        
        Returns:
            (integral, error_estimate)
        """
        def simpson(f, a, b):
            """Simpson's rule"""
            h = (b - a) / 2
            return h / 3 * (f(a) + 4*f(a + h) + f(b))
        
        def adaptive_simpson(f, a, b, tol, whole, depth):
            """Recursive adaptive Simpson"""
            c = (a + b) / 2
            left = simpson(f, a, c)
            right = simpson(f, c, b)
            
            if depth <= 0:
                warnings.warn("Maximum recursion depth reached")
                return left + right
            
            if abs(left + right - whole) < 15 * tol:
                return left + right + (left + right - whole) / 15
            
            return (adaptive_simpson(f, a, c, tol/2, left, depth-1) +
                   adaptive_simpson(f, c, b, tol/2, right, depth-1))
        
        whole = simpson(f, a, b)
        result = adaptive_simpson(f, a, b, tol, whole, max_depth)
        
        # Error estimate using Richardson extrapolation
        h1 = (b - a) / 4
        h2 = (b - a) / 8
        I1 = sum(simpson(f, a + i*h1, a + (i+1)*h1) for i in range(4))
        I2 = sum(simpson(f, a + i*h2, a + (i+1)*h2) for i in range(8))
        error = abs(I2 - I1) / 15
        
        return result, error
    
    def monte_carlo_integration(self, f: Callable, domain: List[Tuple[float, float]],
                              n_samples: int = 100000, 
                              method: str = 'uniform') -> Tuple[float, float]:
        """
        Monte Carlo integration in arbitrary dimensions
        
        Args:
            f: Function to integrate
            domain: List of (min, max) for each dimension
            n_samples: Number of samples
            method: 'uniform' or 'importance' or 'quasi'
        """
        dim = len(domain)
        
        if method == 'uniform':
            # Uniform random sampling
            samples = np.random.uniform(size=(n_samples, dim))
            for i, (a, b) in enumerate(domain):
                samples[:, i] = a + (b - a) * samples[:, i]
            
        elif method == 'quasi':
            # Quasi-random (Sobol sequence)
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random(n_samples)
            for i, (a, b) in enumerate(domain):
                samples[:, i] = a + (b - a) * samples[:, i]
        
        else:  # importance sampling
            # Use multivariate normal centered in domain
            center = np.array([(a + b) / 2 for a, b in domain])
            cov = np.diag([(b - a)**2 / 16 for a, b in domain])
            samples = np.random.multivariate_normal(center, cov, n_samples)
        
        # Evaluate function
        values = np.array([f(*sample) for sample in samples])
        
        # Compute volume
        volume = np.prod([b - a for a, b in domain])
        
        # Estimate integral
        integral = volume * np.mean(values)
        
        # Error estimate (standard error)
        std_error = volume * np.std(values) / np.sqrt(n_samples)
        
        return integral, std_error
    
    def finite_difference_weights(self, x0: float, x: np.ndarray, 
                                n: int, m: int) -> np.ndarray:
        """
        Calculate finite difference weights for arbitrary stencils
        
        Args:
            x0: Point at which to evaluate derivative
            x: Stencil points
            n: Derivative order
            m: Maximum derivative order to compute
        """
        c = np.zeros((len(x), m + 1))
        c1 = 1.0
        c4 = x[0] - x0
        
        for i in range(len(x)):
            mn = min(i, m)
            c2 = 1.0
            c5 = c4
            c4 = x[i] - x0
            
            for j in range(i):
                c3 = x[i] - x[j]
                c2 *= c3
                
                for k in range(mn, -1, -1):
                    c[i, k] = (c4 * c[i-1, k] - k * c[i-1, k-1]) / c3
            
            for k in range(mn, -1, -1):
                c[j, k] = (c2 / c1) * ((k * c[j, k-1] - c5 * c[j, k]) / c3)
            
            c1 = c2
        
        return c[:, n]
    
    def chebyshev_interpolation(self, f: Callable, a: float, b: float, 
                              n: int) -> Tuple[np.ndarray, Callable]:
        """
        Chebyshev polynomial interpolation
        
        Returns:
            (coefficients, interpolation_function)
        """
        # Chebyshev nodes
        k = np.arange(n)
        x = np.cos((2*k + 1) * np.pi / (2*n))
        
        # Transform to [a, b]
        x_scaled = 0.5 * (b - a) * x + 0.5 * (b + a)
        
        # Function values
        y = np.array([f(xi) for xi in x_scaled])
        
        # Compute Chebyshev coefficients using DCT
        c = np.real(fft(y))
        c[0] /= n
        c[1:] *= 2/n
        
        def interpolant(x_eval):
            """Evaluate Chebyshev interpolant"""
            # Transform to [-1, 1]
            t = 2 * (x_eval - a) / (b - a) - 1
            
            # Clenshaw's algorithm
            b2 = 0
            b1 = 0
            for k in range(n-1, 0, -1):
                b0 = c[k] + 2*t*b1 - b2
                b2 = b1
                b1 = b0
            
            return c[0] + t*b1 - b2
        
        return c[:n], interpolant
    
    def runge_kutta_adaptive(self, f: Callable, t_span: Tuple[float, float],
                           y0: np.ndarray, rtol: float = 1e-6,
                           atol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive Runge-Kutta-Fehlberg (RKF45) method
        """
        # Butcher tableau for RKF45
        c = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
        a = np.array([
            [0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40]
        ])
        b = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
        b_star = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        
        t0, tf = t_span
        t = t0
        y = y0.copy()
        
        t_vals = [t]
        y_vals = [y.copy()]
        
        h = 0.01 * (tf - t0)  # Initial step size
        
        while t < tf:
            # RK stages
            k = np.zeros((6, len(y)))
            k[0] = f(t, y)
            
            for i in range(1, 6):
                y_stage = y + h * sum(a[i, j] * k[j] for j in range(i))
                k[i] = f(t + c[i] * h, y_stage)
            
            # Solutions
            y_new = y + h * sum(b[i] * k[i] for i in range(6))
            y_new_star = y + h * sum(b_star[i] * k[i] for i in range(6))
            
            # Error estimate
            error = np.linalg.norm(y_new - y_new_star)
            tol = atol + rtol * np.linalg.norm(y)
            
            if error <= tol:
                # Accept step
                t += h
                y = y_new_star
                t_vals.append(t)
                y_vals.append(y.copy())
            
            # Adjust step size
            if error > 0:
                h_new = 0.9 * h * (tol / error) ** 0.2
                h = max(0.1 * h, min(5 * h, h_new))
            
            # Ensure we don't overshoot
            if t + h > tf:
                h = tf - t
        
        return np.array(t_vals), np.array(y_vals)
    
    def nonlinear_conjugate_gradient(self, f: Callable, grad_f: Callable,
                                   x0: np.ndarray, max_iter: int = 1000,
                                   tol: float = 1e-8) -> Dict[str, Any]:
        """
        Nonlinear conjugate gradient method (Fletcher-Reeves)
        """
        x = x0.copy()
        g = grad_f(x)
        d = -g
        
        history = {'f_vals': [f(x)], 'grad_norms': [np.linalg.norm(g)]}
        
        for k in range(max_iter):
            if np.linalg.norm(g) < tol:
                break
            
            # Line search
            def phi(alpha):
                return f(x + alpha * d)
            
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(phi, bounds=(0, 1), method='bounded')
            alpha = result.x
            
            # Update
            x_new = x + alpha * d
            g_new = grad_f(x_new)
            
            # Fletcher-Reeves parameter
            beta = np.dot(g_new, g_new) / np.dot(g, g)
            
            # Update direction
            d = -g_new + beta * d
            
            # Move to next iteration
            x = x_new
            g = g_new
            
            history['f_vals'].append(f(x))
            history['grad_norms'].append(np.linalg.norm(g))
        
        return {
            'x': x,
            'f_val': f(x),
            'iterations': k + 1,
            'history': history,
            'converged': np.linalg.norm(g) < tol
        }
    
    def sparse_grid_quadrature(self, f: Callable, dim: int, level: int,
                             domain: Optional[List[Tuple[float, float]]] = None) -> float:
        """
        Sparse grid quadrature (Smolyak) for high-dimensional integration
        """
        if domain is None:
            domain = [(-1, 1)] * dim
        
        # Gauss-Legendre nodes and weights for 1D
        def gauss_legendre_1d(n):
            from numpy.polynomial.legendre import leggauss
            return leggauss(n)
        
        # Smolyak sparse grid construction
        total = 0
        
        # This is a simplified version - full implementation would be more complex
        for k in range(1, level + 1):
            nodes_1d, weights_1d = gauss_legendre_1d(2**k + 1)
            
            # Transform to domain
            for i, (a, b) in enumerate(domain):
                nodes_1d = 0.5 * (b - a) * nodes_1d + 0.5 * (b + a)
                weights_1d *= 0.5 * (b - a)
            
            # Tensor product for this level
            grid = np.meshgrid(*[nodes_1d] * dim)
            points = np.stack([g.flatten() for g in grid], axis=1)
            
            # Evaluate function
            values = np.array([f(*p) for p in points])
            
            # Combine with weights
            weight_grid = np.meshgrid(*[weights_1d] * dim)
            weights = np.prod(np.stack([w.flatten() for w in weight_grid], axis=1), axis=1)
            
            # Smolyak combination coefficient
            coeff = (-1)**(level - k) * comb(dim - 1, level - k, exact=True)
            
            total += coeff * np.dot(weights, values)
        
        return total