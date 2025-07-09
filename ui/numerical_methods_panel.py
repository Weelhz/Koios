import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core.numerical_methods_engine import NumericalMethodsEngine, SolverType

def render_numerical_methods_panel():
    """Render the numerical methods interface"""
    st.header("🔢 Advanced Numerical Methods")
    
    tabs = st.tabs([
        "PDE Solvers",
        "Integration",
        "ODE Solvers", 
        "Interpolation",
        "Optimization"
    ])
    
    with tabs[0]:
        render_pde_solvers()
    
    with tabs[1]:
        render_integration_methods()
        
    with tabs[2]:
        render_ode_solvers()
        
    with tabs[3]:
        render_interpolation()
        
    with tabs[4]:
        render_optimization_methods()

def render_pde_solvers():
    """PDE solving methods"""
    st.subheader("Partial Differential Equation Solvers")
    
    solver_type = st.selectbox(
        "Select solver type",
        ["Spectral Poisson Solver", "Multigrid Method", "FDTD Electromagnetic"]
    )
    
    if solver_type == "Spectral Poisson Solver":
        st.write("**2D Poisson Equation: ∇²u = f**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Domain settings
            st.write("**Domain Settings**")
            x_min = st.number_input("x_min", value=0.0)
            x_max = st.number_input("x_max", value=1.0)
            y_min = st.number_input("y_min", value=0.0)
            y_max = st.number_input("y_max", value=1.0)
            
            nx = st.slider("Grid points (x)", 16, 128, 64)
            ny = st.slider("Grid points (y)", 16, 128, 64)
            
            bc_type = st.selectbox("Boundary conditions", ["dirichlet", "periodic"])
        
        with col2:
            # Source function
            st.write("**Source Function f(x,y)**")
            source_type = st.selectbox(
                "Source type",
                ["Gaussian", "Sinusoidal", "Point sources", "Custom"]
            )
            
            if source_type == "Custom":
                custom_f = st.text_area(
                    "Enter f(x,y) expression",
                    value="np.sin(2*np.pi*x) * np.sin(2*np.pi*y)"
                )
        
        if st.button("Solve Poisson Equation"):
            try:
                engine = NumericalMethodsEngine()
                
                # Create grid
                x = np.linspace(x_min, x_max, nx)
                y = np.linspace(y_min, y_max, ny)
                X, Y = np.meshgrid(x, y)
                
                # Define source function
                if source_type == "Gaussian":
                    f = 10 * np.exp(-((X-0.5)**2 + (Y-0.5)**2) / 0.05)
                elif source_type == "Sinusoidal":
                    f = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
                elif source_type == "Point sources":
                    f = np.zeros((ny, nx))
                    f[ny//4, nx//4] = 100
                    f[3*ny//4, 3*nx//4] = -100
                else:  # Custom
                    x, y = X, Y  # For eval
                    f = eval(custom_f)
                
                # Solve
                with st.spinner("Solving..."):
                    u = engine.spectral_poisson_solver_2d(
                        f, (x_min, x_max, y_min, y_max), bc_type
                    )
                
                # Visualize
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(6, 5))
                    im1 = ax1.contourf(X, Y, f, levels=20, cmap='RdBu')
                    ax1.set_title('Source f(x,y)')
                    ax1.set_xlabel('x')
                    ax1.set_ylabel('y')
                    plt.colorbar(im1, ax=ax1)
                    st.pyplot(fig1)
                
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(6, 5))
                    im2 = ax2.contourf(X, Y, u, levels=20, cmap='viridis')
                    ax2.set_title('Solution u(x,y)')
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('y')
                    plt.colorbar(im2, ax=ax2)
                    st.pyplot(fig2)
                
                # 3D visualization
                if st.checkbox("Show 3D plot"):
                    fig3 = plt.figure(figsize=(10, 8))
                    ax3 = fig3.add_subplot(111, projection='3d')
                    ax3.plot_surface(X, Y, u, cmap='viridis', alpha=0.8)
                    ax3.set_xlabel('x')
                    ax3.set_ylabel('y')
                    ax3.set_zlabel('u(x,y)')
                    ax3.set_title('Solution Surface')
                    st.pyplot(fig3)
                
            except Exception as e:
                st.error(f"Error solving PDE: {str(e)}")
    
    elif solver_type == "Multigrid Method":
        st.write("**Multigrid V-Cycle Solver**")
        st.info("Solves linear system Ax = b using multigrid method")
        
        # Matrix size
        n = st.slider("System size", 16, 256, 64)
        
        # Problem type
        problem = st.selectbox(
            "Test problem",
            ["1D Poisson", "2D Laplacian", "Custom matrix"]
        )
        
        if st.button("Solve with Multigrid"):
            try:
                from scipy.sparse import diags
                engine = NumericalMethodsEngine()
                
                # Create test problem
                if problem == "1D Poisson":
                    # Tridiagonal matrix
                    A = diags([-1, 2, -1], [-1, 0, 1], shape=(n, n), format='csr')
                    b = np.ones(n) / n**2
                elif problem == "2D Laplacian":
                    # 2D finite difference Laplacian
                    n_sqrt = int(np.sqrt(n))
                    n = n_sqrt**2
                    main_diag = 4 * np.ones(n)
                    side_diag = -np.ones(n-1)
                    side_diag[n_sqrt-1::n_sqrt] = 0
                    up_diag = -np.ones(n-n_sqrt)
                    
                    A = diags([up_diag, side_diag, main_diag, side_diag, up_diag],
                             [-n_sqrt, -1, 0, 1, n_sqrt], format='csr')
                    b = np.random.randn(n)
                else:
                    st.warning("Custom matrix not implemented in demo")
                    return
                
                x0 = np.zeros(n)
                
                # Solve
                with st.spinner("Running multigrid..."):
                    x = engine.multigrid_v_cycle(A, b, x0, levels=3)
                
                # Check residual
                residual = np.linalg.norm(A @ x - b)
                
                st.success(f"Solution found! Residual: {residual:.2e}")
                
                # Visualize for 2D case
                if problem == "2D Laplacian":
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                    
                    # Reshape for 2D
                    b_2d = b.reshape(n_sqrt, n_sqrt)
                    x_2d = x.reshape(n_sqrt, n_sqrt)
                    
                    im1 = ax1.imshow(b_2d, cmap='RdBu')
                    ax1.set_title('Right-hand side b')
                    plt.colorbar(im1, ax=ax1)
                    
                    im2 = ax2.imshow(x_2d, cmap='viridis')
                    ax2.set_title('Solution x')
                    plt.colorbar(im2, ax=ax2)
                    
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error in multigrid solver: {str(e)}")

def render_integration_methods():
    """Numerical integration methods"""
    st.subheader("Numerical Integration")
    
    method = st.selectbox(
        "Integration method",
        ["Adaptive Quadrature", "Monte Carlo", "Sparse Grid (High-D)"]
    )
    
    if method == "Adaptive Quadrature":
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Function Settings**")
            func_type = st.selectbox(
                "Test function",
                ["exp(-x²)", "1/(1+x²)", "sin(x)/x", "Oscillatory", "Custom"]
            )
            
            a = st.number_input("Lower limit a", value=0.0)
            b = st.number_input("Upper limit b", value=1.0)
            tol = st.number_input("Tolerance", value=1e-8, format="%.2e")
            
            if func_type == "Custom":
                custom_func = st.text_input("f(x) =", value="np.exp(-x**2)")
        
        with col2:
            if st.button("Integrate"):
                try:
                    engine = NumericalMethodsEngine()
                    
                    # Define function
                    if func_type == "exp(-x²)":
                        f = lambda x: np.exp(-x**2)
                        exact = np.sqrt(np.pi) * (np.math.erf(b) - np.math.erf(a)) / 2
                    elif func_type == "1/(1+x²)":
                        f = lambda x: 1 / (1 + x**2)
                        exact = np.arctan(b) - np.arctan(a)
                    elif func_type == "sin(x)/x":
                        f = lambda x: np.sin(x) / x if x != 0 else 1
                        exact = None  # No simple form
                    elif func_type == "Oscillatory":
                        f = lambda x: np.sin(100 * x)
                        exact = (np.cos(100*a) - np.cos(100*b)) / 100
                    else:  # Custom
                        f = lambda x: eval(custom_func)
                        exact = None
                    
                    # Integrate
                    result, error = engine.adaptive_integration(f, a, b, tol)
                    
                    st.write("**Results:**")
                    st.write(f"Integral: {result:.10f}")
                    st.write(f"Error estimate: {error:.2e}")
                    
                    if exact is not None:
                        true_error = abs(result - exact)
                        st.write(f"Exact value: {exact:.10f}")
                        st.write(f"True error: {true_error:.2e}")
                    
                    # Plot function
                    x = np.linspace(a, b, 1000)
                    y = [f(xi) for xi in x]
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(x, y, 'b-', label='f(x)')
                    ax.fill_between(x, 0, y, alpha=0.3)
                    ax.set_xlabel('x')
                    ax.set_ylabel('f(x)')
                    ax.set_title(f'Adaptive Integration: ∫f(x)dx = {result:.6f}')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Integration error: {str(e)}")
    
    elif method == "Monte Carlo":
        st.write("**Monte Carlo Integration**")
        
        dim = st.slider("Dimensions", 1, 10, 2)
        n_samples = st.number_input("Number of samples", 1000, 10000000, 100000)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Domain**")
            domain = []
            for i in range(dim):
                col_a, col_b = st.columns(2)
                with col_a:
                    a = st.number_input(f"x{i} min", value=0.0, key=f"mc_a_{i}")
                with col_b:
                    b = st.number_input(f"x{i} max", value=1.0, key=f"mc_b_{i}")
                domain.append((a, b))
        
        with col2:
            st.write("**Function**")
            if dim == 2:
                func_type = st.selectbox(
                    "Test function",
                    ["Unit sphere", "Gaussian", "Polynomial", "Custom"]
                )
            else:
                func_type = "Gaussian"
            
            mc_method = st.selectbox("Sampling", ["uniform", "quasi"])
        
        if st.button("Run Monte Carlo"):
            try:
                engine = NumericalMethodsEngine()
                
                # Define function
                if func_type == "Unit sphere" and dim == 2:
                    f = lambda x, y: 1 if x**2 + y**2 <= 1 else 0
                    exact = np.pi / 4 if domain == [(0,1), (0,1)] else None
                elif func_type == "Gaussian":
                    f = lambda *args: np.exp(-sum(x**2 for x in args))
                    exact = (np.sqrt(np.pi))**dim if all(d == (-5, 5) for d in domain) else None
                elif func_type == "Polynomial" and dim == 2:
                    f = lambda x, y: x**2 + y**2
                    exact = 2/3 if domain == [(0,1), (0,1)] else None
                else:
                    f = lambda *args: np.exp(-sum(x**2 for x in args))
                    exact = None
                
                # Integrate
                with st.spinner(f"Running {n_samples:,} samples..."):
                    result, error = engine.monte_carlo_integration(
                        f, domain, n_samples, mc_method
                    )
                
                st.write("**Results:**")
                st.write(f"Integral: {result:.6f} ± {error:.6f}")
                st.write(f"Relative error: {error/abs(result)*100:.2f}%")
                
                if exact is not None:
                    st.write(f"Exact value: {exact:.6f}")
                    st.write(f"True error: {abs(result - exact):.6f}")
                
                # Convergence plot
                if st.checkbox("Show convergence"):
                    n_points = 10
                    samples = np.logspace(3, np.log10(n_samples), n_points, dtype=int)
                    results = []
                    errors = []
                    
                    progress = st.progress(0)
                    for i, n in enumerate(samples):
                        r, e = engine.monte_carlo_integration(f, domain, n, mc_method)
                        results.append(r)
                        errors.append(e)
                        progress.progress((i+1)/n_points)
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.errorbar(samples, results, yerr=errors, fmt='o-', capsize=5)
                    if exact is not None:
                        ax.axhline(exact, color='r', linestyle='--', label='Exact')
                    ax.set_xscale('log')
                    ax.set_xlabel('Number of samples')
                    ax.set_ylabel('Integral value')
                    ax.set_title('Monte Carlo Convergence')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.pyplot(fig)
                    
            except Exception as e:
                st.error(f"Monte Carlo error: {str(e)}")

def render_ode_solvers():
    """ODE solving methods"""
    st.subheader("Advanced ODE Solvers")
    
    st.write("**Adaptive Runge-Kutta-Fehlberg (RKF45)**")
    
    # ODE selection
    ode_type = st.selectbox(
        "Select ODE system",
        ["Van der Pol oscillator", "Lorenz system", "Pendulum", "Custom"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if ode_type == "Van der Pol oscillator":
            mu = st.slider("μ parameter", 0.1, 5.0, 1.0)
            y0 = [st.number_input("y₀", value=2.0), 
                  st.number_input("y'₀", value=0.0)]
            
            def f(t, y):
                return np.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])
                
        elif ode_type == "Lorenz system":
            sigma = st.number_input("σ", value=10.0)
            rho = st.number_input("ρ", value=28.0)
            beta = st.number_input("β", value=8/3)
            y0 = [st.number_input("x₀", value=1.0),
                  st.number_input("y₀", value=1.0),
                  st.number_input("z₀", value=1.0)]
            
            def f(t, y):
                return np.array([
                    sigma * (y[1] - y[0]),
                    y[0] * (rho - y[2]) - y[1],
                    y[0] * y[1] - beta * y[2]
                ])
        else:
            st.info("Custom ODE: dy/dt = f(t, y)")
            y0 = [1.0]
            def f(t, y):
                return np.array([-y[0]])
    
    with col2:
        t0 = st.number_input("Initial time", value=0.0)
        tf = st.number_input("Final time", value=10.0)
        rtol = st.number_input("Relative tolerance", value=1e-6, format="%.2e")
        atol = st.number_input("Absolute tolerance", value=1e-9, format="%.2e")
    
    if st.button("Solve ODE"):
        try:
            engine = NumericalMethodsEngine()
            y0 = np.array(y0)
            
            with st.spinner("Solving..."):
                t, y = engine.runge_kutta_adaptive(f, (t0, tf), y0, rtol, atol)
            
            st.success(f"Solution computed with {len(t)} time steps")
            
            # Plot results
            if len(y0) == 1:
                # 1D system
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(t, y[:, 0])
                ax.set_xlabel('t')
                ax.set_ylabel('y(t)')
                ax.set_title('ODE Solution')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
            elif len(y0) == 2:
                # 2D system - phase portrait
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Time series
                ax1.plot(t, y[:, 0], label='y₁')
                ax1.plot(t, y[:, 1], label='y₂')
                ax1.set_xlabel('t')
                ax1.set_ylabel('y')
                ax1.set_title('Time Series')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Phase portrait
                ax2.plot(y[:, 0], y[:, 1])
                ax2.plot(y[0, 0], y[0, 1], 'go', markersize=8, label='Start')
                ax2.plot(y[-1, 0], y[-1, 1], 'ro', markersize=8, label='End')
                ax2.set_xlabel('y₁')
                ax2.set_ylabel('y₂')
                ax2.set_title('Phase Portrait')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
            else:
                # 3D system
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(y[:, 0], y[:, 1], y[:, 2])
                ax.scatter(y[0, 0], y[0, 1], y[0, 2], c='g', s=100, label='Start')
                ax.scatter(y[-1, 0], y[-1, 1], y[-1, 2], c='r', s=100, label='End')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.set_title('3D Trajectory')
                ax.legend()
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"ODE solver error: {str(e)}")

def render_interpolation():
    """Interpolation methods"""
    st.subheader("Chebyshev Interpolation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Function to interpolate**")
        func_type = st.selectbox(
            "Function",
            ["Runge function", "Step function", "Oscillatory", "Custom"]
        )
        
        a = st.number_input("Left endpoint", value=-1.0)
        b = st.number_input("Right endpoint", value=1.0)
        n = st.slider("Number of Chebyshev nodes", 5, 50, 15)
        
        if func_type == "Custom":
            custom_f = st.text_input("f(x) =", value="np.sin(5*x)")
    
    with col2:
        if st.button("Compute Interpolation"):
            try:
                engine = NumericalMethodsEngine()
                
                # Define function
                if func_type == "Runge function":
                    f = lambda x: 1 / (1 + 25 * x**2)
                elif func_type == "Step function":
                    f = lambda x: 1 if x > 0 else -1
                elif func_type == "Oscillatory":
                    f = lambda x: np.sin(10 * x) * np.exp(-x**2)
                else:
                    f = lambda x: eval(custom_f)
                
                # Compute interpolation
                coeffs, interpolant = engine.chebyshev_interpolation(f, a, b, n)
                
                # Evaluate on fine grid
                x_fine = np.linspace(a, b, 500)
                y_true = [f(x) for x in x_fine]
                y_interp = [interpolant(x) for x in x_fine]
                
                # Chebyshev nodes
                k = np.arange(n)
                x_cheb = np.cos((2*k + 1) * np.pi / (2*n))
                x_cheb_scaled = 0.5 * (b - a) * x_cheb + 0.5 * (b + a)
                y_cheb = [f(x) for x in x_cheb_scaled]
                
                # Plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
                
                # Interpolation
                ax1.plot(x_fine, y_true, 'b-', label='True function', linewidth=2)
                ax1.plot(x_fine, y_interp, 'r--', label='Chebyshev interpolant', linewidth=2)
                ax1.plot(x_cheb_scaled, y_cheb, 'ko', markersize=8, label='Chebyshev nodes')
                ax1.set_xlabel('x')
                ax1.set_ylabel('f(x)')
                ax1.set_title(f'Chebyshev Interpolation (n={n})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Error
                error = np.array(y_interp) - np.array(y_true)
                ax2.semilogy(x_fine, np.abs(error))
                ax2.set_xlabel('x')
                ax2.set_ylabel('|Error|')
                ax2.set_title('Interpolation Error')
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Coefficient decay
                if st.checkbox("Show coefficient decay"):
                    fig2, ax = plt.subplots(figsize=(8, 5))
                    ax.semilogy(np.abs(coeffs), 'o-')
                    ax.set_xlabel('Coefficient index')
                    ax.set_ylabel('|Coefficient|')
                    ax.set_title('Chebyshev Coefficient Decay')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                    
                    st.info("""
                    Fast coefficient decay indicates smooth function.
                    Slow decay suggests discontinuities or sharp features.
                    """)
                
            except Exception as e:
                st.error(f"Interpolation error: {str(e)}")

def render_optimization_methods():
    """Optimization methods from numerical perspective"""
    st.subheader("Nonlinear Conjugate Gradient")
    
    st.write("**Minimize f(x) using Fletcher-Reeves conjugate gradient**")
    
    # Test functions
    func_type = st.selectbox(
        "Test function",
        ["Rosenbrock", "Quadratic", "Booth function", "Custom"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        dim = st.slider("Dimensions", 2, 10, 2)
        x0 = []
        st.write("**Initial point**")
        for i in range(dim):
            x0.append(st.number_input(f"x₀[{i}]", value=0.0, key=f"x0_{i}"))
        x0 = np.array(x0)
        
        max_iter = st.number_input("Max iterations", 100, 10000, 1000)
        tol = st.number_input("Tolerance", value=1e-8, format="%.2e")
    
    with col2:
        if func_type == "Rosenbrock":
            def f(x):
                return sum(100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2 
                          for i in range(len(x)-1))
            def grad_f(x):
                g = np.zeros_like(x)
                for i in range(len(x)-1):
                    g[i] += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1-x[i])
                    g[i+1] += 200*(x[i+1] - x[i]**2)
                return g
                
        elif func_type == "Quadratic":
            # Random positive definite quadratic
            np.random.seed(42)
            A = np.random.randn(dim, dim)
            A = A.T @ A + 0.1 * np.eye(dim)  # Make positive definite
            b = np.random.randn(dim)
            
            def f(x):
                return 0.5 * x @ A @ x - b @ x
            def grad_f(x):
                return A @ x - b
        else:
            st.warning("Please select a predefined function")
            return
    
    if st.button("Optimize"):
        try:
            engine = NumericalMethodsEngine()
            
            with st.spinner("Optimizing..."):
                result = engine.nonlinear_conjugate_gradient(
                    f, grad_f, x0, max_iter, tol
                )
            
            st.write("**Results:**")
            st.write(f"Optimal point: {result['x']}")
            st.write(f"Optimal value: {result['f_val']:.6f}")
            st.write(f"Iterations: {result['iterations']}")
            st.write(f"Converged: {result['converged']}")
            st.write(f"Final gradient norm: {result['history']['grad_norms'][-1]:.2e}")
            
            # Convergence plots
            history = result['history']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Function values
            ax1.semilogy(history['f_vals'])
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('f(x)')
            ax1.set_title('Function Value Convergence')
            ax1.grid(True, alpha=0.3)
            
            # Gradient norms
            ax2.semilogy(history['grad_norms'])
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('||∇f||')
            ax2.set_title('Gradient Norm Convergence')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")