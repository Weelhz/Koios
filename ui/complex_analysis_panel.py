import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import sys
from pathlib import Path

# Add core directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / "core"))

try:
    from complex_analysis_engine import ComplexAnalysisEngine
except ImportError:
    st.error("Failed to import ComplexAnalysisEngine")
    ComplexAnalysisEngine = None

def render_complex_analysis_panel():
    """Render the complex analysis panel"""
    st.header("Complex Analysis Tools")
    st.markdown("Advanced complex number calculations and analysis")
    
    if ComplexAnalysisEngine is None:
        st.error("Complex Analysis Engine not available. Please check installation.")
        return
    
    # Initialize engine
    engine = ComplexAnalysisEngine()
    
    # Create tabs for different complex analysis operations
    tabs = st.tabs([
        "Basic Operations",
        "Complex Functions",
        "Contour Integration",
        "Conformal Mappings",
        "Laurent Series",
        "Visualization"
    ])
    
    with tabs[0]:
        render_basic_operations(engine)
    
    with tabs[1]:
        render_complex_functions(engine)
    
    with tabs[2]:
        render_contour_integration(engine)
    
    with tabs[3]:
        render_conformal_mappings(engine)
    
    with tabs[4]:
        render_laurent_series(engine)
    
    with tabs[5]:
        render_visualization(engine)

def render_basic_operations(engine: ComplexAnalysisEngine):
    """Render basic complex operations tab"""
    st.subheader("Basic Complex Operations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**First Complex Number (z1)**")
        z1_real = st.number_input("Real part", key="z1_real", value=3.0, step=0.1)
        z1_imag = st.number_input("Imaginary part", key="z1_imag", value=4.0, step=0.1)
        z1 = complex(z1_real, z1_imag)
    
    with col2:
        st.markdown("**Second Complex Number (z2)**")
        z2_real = st.number_input("Real part", key="z2_real", value=1.0, step=0.1)
        z2_imag = st.number_input("Imaginary part", key="z2_imag", value=-2.0, step=0.1)
        z2 = complex(z2_real, z2_imag)
    
    if st.button("Calculate", key="basic_calc"):
        try:
            # Arithmetic operations
            arithmetic = engine.complex_arithmetic(z1, z2)
            
            # Display results
            st.markdown("### Results")
            
            # Format complex numbers for display
            results_data = []
            for op, result in arithmetic.items():
                if result is not None:
                    if isinstance(result, complex):
                        formatted = f"{result.real:.4f} + {result.imag:.4f}i" if result.imag >= 0 else f"{result.real:.4f} - {abs(result.imag):.4f}i"
                    else:
                        formatted = f"{result:.4f}"
                    results_data.append([op.replace('_', ' ').title(), formatted])
            
            results_df = pd.DataFrame(results_data, columns=["Operation", "Result"])
            st.dataframe(results_df, use_container_width=True)
            
            # Polar and exponential forms
            st.markdown("### Alternative Forms")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**z1 Polar Form**")
                polar1 = engine.polar_form(z1)
                st.write(f"r = {polar1['r']:.4f}")
                st.write(f"θ = {polar1['theta']:.4f} rad ({polar1['theta_degrees']:.2f}°)")
            
            with col2:
                st.markdown("**z2 Polar Form**")
                polar2 = engine.polar_form(z2)
                st.write(f"r = {polar2['r']:.4f}")
                st.write(f"θ = {polar2['theta']:.4f} rad ({polar2['theta_degrees']:.2f}°)")
            
        except Exception as e:
            st.error(f"Error in calculations: {str(e)}")
    
    # Complex roots section
    st.markdown("---")
    st.subheader("Complex Roots")
    
    col1, col2 = st.columns(2)
    
    with col1:
        root_real = st.number_input("Real part", key="root_real", value=-8.0, step=0.1)
        root_imag = st.number_input("Imaginary part", key="root_imag", value=0.0, step=0.1)
        z_root = complex(root_real, root_imag)
    
    with col2:
        n_root = st.number_input("Root degree (n)", key="n_root", value=3, min_value=2, max_value=10)
    
    if st.button("Find Roots", key="find_roots"):
        try:
            roots = engine.complex_roots(z_root, n_root)
            
            st.markdown(f"### {n_root}th roots of {z_root}")
            
            roots_data = []
            for i, root in enumerate(roots):
                formatted = f"{root.real:.4f} + {root.imag:.4f}i" if root.imag >= 0 else f"{root.real:.4f} - {abs(root.imag):.4f}i"
                roots_data.append([f"Root {i+1}", formatted])
            
            roots_df = pd.DataFrame(roots_data, columns=["Root", "Value"])
            st.dataframe(roots_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error finding roots: {str(e)}")

def render_complex_functions(engine: ComplexAnalysisEngine):
    """Render complex functions tab"""
    st.subheader("Complex Functions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        func_real = st.number_input("Real part", key="func_real", value=1.0, step=0.1)
        func_imag = st.number_input("Imaginary part", key="func_imag", value=1.0, step=0.1)
        z_func = complex(func_real, func_imag)
    
    if st.button("Evaluate Functions", key="eval_func"):
        try:
            functions = engine.analytic_functions(z_func)
            
            st.markdown(f"### Function values at z = {z_func}")
            
            func_data = []
            for func_name, value in functions.items():
                if value is not None:
                    if isinstance(value, complex):
                        formatted = f"{value.real:.4f} + {value.imag:.4f}i" if value.imag >= 0 else f"{value.real:.4f} - {abs(value.imag):.4f}i"
                    else:
                        formatted = f"{value:.4f}"
                    func_data.append([func_name, formatted])
            
            func_df = pd.DataFrame(func_data, columns=["Function", "Value"])
            st.dataframe(func_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error evaluating functions: {str(e)}")
    
    # Complex logarithm section
    st.markdown("---")
    st.subheader("Complex Logarithm")
    
    col1, col2 = st.columns(2)
    
    with col1:
        log_real = st.number_input("Real part", key="log_real", value=1.0, step=0.1)
        log_imag = st.number_input("Imaginary part", key="log_imag", value=0.0, step=0.1)
        z_log = complex(log_real, log_imag)
    
    with col2:
        branch = st.number_input("Branch number", key="branch", value=0, min_value=-5, max_value=5)
    
    if st.button("Calculate Logarithm", key="calc_log"):
        try:
            log_value = engine.complex_logarithm(z_log, branch)
            
            st.markdown(f"### log(z) for branch {branch}")
            if isinstance(log_value, complex):
                formatted = f"{log_value.real:.4f} + {log_value.imag:.4f}i" if log_value.imag >= 0 else f"{log_value.real:.4f} - {abs(log_value.imag):.4f}i"
            else:
                formatted = f"{log_value:.4f}"
            
            st.success(f"log({z_log}) = {formatted}")
            
        except Exception as e:
            st.error(f"Error calculating logarithm: {str(e)}")

def render_contour_integration(engine: ComplexAnalysisEngine):
    """Render contour integration tab"""
    st.subheader("Contour Integration")
    
    st.info("Calculate contour integrals using numerical methods")
    
    # Predefined examples
    example = st.selectbox(
        "Select example",
        ["Custom", "1/z around unit circle", "z² around unit circle", "sin(z)/z around unit circle"]
    )
    
    if example == "Custom":
        function_str = st.text_input("Function f(z)", value="1/z", help="Enter function in terms of z")
        contour_type = st.selectbox("Contour type", ["Circle", "Line segment", "Rectangle"])
        
        if contour_type == "Circle":
            col1, col2 = st.columns(2)
            with col1:
                center_real = st.number_input("Center (real)", value=0.0, step=0.1)
                center_imag = st.number_input("Center (imag)", value=0.0, step=0.1)
            with col2:
                radius = st.number_input("Radius", value=1.0, min_value=0.1, step=0.1)
        
        if st.button("Calculate Integral", key="calc_integral"):
            try:
                # Create function
                def f(z):
                    return eval(function_str.replace('z', f'({z})'))
                
                # Create contour
                if contour_type == "Circle":
                    center = complex(center_real, center_imag)
                    def contour(t):
                        return center + radius * np.exp(2j * np.pi * t)
                
                # Calculate integral
                integral = engine.contour_integral(f, contour)
                
                st.markdown("### Result")
                formatted = f"{integral.real:.6f} + {integral.imag:.6f}i" if integral.imag >= 0 else f"{integral.real:.6f} - {abs(integral.imag):.6f}i"
                st.success(f"∮ f(z) dz = {formatted}")
                
                # Check if it's 2πi times an integer (residue theorem)
                if abs(integral.real) < 1e-6:
                    n = integral.imag / (2 * np.pi)
                    if abs(n - round(n)) < 1e-6:
                        st.info(f"This equals 2πi × {int(round(n))}, suggesting {int(round(n))} residue(s) inside the contour")
                
            except Exception as e:
                st.error(f"Error calculating integral: {str(e)}")
    
    else:
        # Handle predefined examples
        st.info(f"Calculating: {example}")
        if st.button("Calculate", key="calc_example"):
            try:
                if example == "1/z around unit circle":
                    result = 2 * np.pi * 1j
                    st.success(f"∮ 1/z dz = 2πi (by residue theorem)")
                elif example == "z² around unit circle":
                    result = 0
                    st.success(f"∮ z² dz = 0 (analytic function)")
                elif example == "sin(z)/z around unit circle":
                    result = 2 * np.pi * 1j
                    st.success(f"∮ sin(z)/z dz = 2πi (residue at z=0 is 1)")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

def render_conformal_mappings(engine: ComplexAnalysisEngine):
    """Render conformal mappings tab"""
    st.subheader("Conformal Mappings")
    
    mapping_type = st.selectbox(
        "Select mapping",
        ["Joukowski Transform", "Möbius Transform", "Exponential", "Logarithm", "Custom"]
    )
    
    if mapping_type == "Joukowski Transform":
        st.info("w = z + 1/z (used for airfoil shapes)")
        
        test_point = st.text_input("Test point z", value="1+1j")
        
        if st.button("Apply Mapping", key="joukowski"):
            try:
                z = complex(test_point)
                w = engine.joukowski_transform(z)
                
                st.markdown("### Result")
                st.success(f"z = {z} → w = {w:.4f}")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif mapping_type == "Möbius Transform":
        st.info("w = (az + b)/(cz + d)")
        
        col1, col2 = st.columns(2)
        with col1:
            a = complex(st.text_input("a", value="1+0j"))
            b = complex(st.text_input("b", value="0+0j"))
        with col2:
            c = complex(st.text_input("c", value="0+0j"))
            d = complex(st.text_input("d", value="1+0j"))
        
        test_point = st.text_input("Test point z", value="1+1j", key="mobius_test")
        
        if st.button("Apply Mapping", key="mobius"):
            try:
                z = complex(test_point)
                w = engine.mobius_transform(z, a, b, c, d)
                
                st.markdown("### Result")
                st.success(f"z = {z} → w = {w:.4f}")
                
                # Check if it's a valid Möbius transform
                det = a*d - b*c
                if abs(det) < 1e-10:
                    st.warning("Warning: ad - bc ≈ 0, transform is singular")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

def render_laurent_series(engine: ComplexAnalysisEngine):
    """Render Laurent series tab"""
    st.subheader("Laurent Series")
    
    st.info("Compute Laurent series coefficients numerically")
    
    function_str = st.text_input("Function f(z)", value="1/(z*(z-1))", help="Enter function in terms of z")
    
    col1, col2 = st.columns(2)
    with col1:
        center_real = st.number_input("Expansion center (real)", value=0.0, step=0.1)
        center_imag = st.number_input("Expansion center (imag)", value=0.0, step=0.1)
    with col2:
        order = st.number_input("Maximum order", value=5, min_value=1, max_value=20)
    
    if st.button("Compute Series", key="laurent"):
        try:
            center = complex(center_real, center_imag)
            
            # Create function
            def f(z):
                return eval(function_str.replace('z', f'({z})'))
            
            # Compute coefficients
            coefficients = engine.laurent_series(f, center, order)
            
            st.markdown("### Laurent Series Coefficients")
            st.markdown(f"f(z) = Σ c_n (z - {center})^n")
            
            # Display coefficients
            coeff_data = []
            for n in sorted(coefficients.keys()):
                c = coefficients[n]
                if abs(c) > 1e-10:  # Only show non-zero coefficients
                    formatted = f"{c.real:.6f} + {c.imag:.6f}i" if c.imag >= 0 else f"{c.real:.6f} - {abs(c.imag):.6f}i"
                    coeff_data.append([n, formatted])
            
            coeff_df = pd.DataFrame(coeff_data, columns=["n", "c_n"])
            st.dataframe(coeff_df, use_container_width=True)
            
            # Identify poles
            pole_orders = [n for n in coefficients if n < 0 and abs(coefficients[n]) > 1e-10]
            if pole_orders:
                st.info(f"Function has a pole of order {-min(pole_orders)} at z = {center}")
            
        except Exception as e:
            st.error(f"Error computing series: {str(e)}")

def render_visualization(engine: ComplexAnalysisEngine):
    """Render visualization tab"""
    st.subheader("Complex Function Visualization")
    
    st.info("Visualize complex functions using domain coloring")
    
    function_str = st.text_input("Function f(z)", value="z**2", help="Enter function in terms of z")
    
    col1, col2 = st.columns(2)
    with col1:
        x_min = st.number_input("x min", value=-2.0, step=0.5)
        x_max = st.number_input("x max", value=2.0, step=0.5)
    with col2:
        y_min = st.number_input("y min", value=-2.0, step=0.5)
        y_max = st.number_input("y max", value=2.0, step=0.5)
    
    resolution = st.slider("Resolution", min_value=50, max_value=200, value=100, step=10)
    
    if st.button("Visualize", key="viz"):
        try:
            # Create function
            def f(z):
                return eval(function_str.replace('z', f'({z})'))
            
            # Generate visualization
            with st.spinner("Generating visualization..."):
                fig = engine.visualize_complex_function(
                    f,
                    x_range=(x_min, x_max),
                    y_range=(y_min, y_max),
                    n_points=resolution
                )
                
                st.pyplot(fig)
                
            st.info("Left: Argument (phase) of f(z) - colors represent different angles")
            st.info("Right: Modulus (magnitude) of f(z) - brightness represents magnitude")
            
        except Exception as e:
            st.error(f"Error visualizing function: {str(e)}")

if __name__ == "__main__":
    render_complex_analysis_panel()