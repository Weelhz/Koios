import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from core.calculus_engine import calculus_engine
from core.expression_parser import expression_parser
from core.advanced_ode_solver import AdvancedODESolver, ODEProblem, ODEType, SolverMethod

def render_calculus_panel():
    """Render the calculus tools panel"""
    st.header("Calculus Tools")
    st.markdown("*Comprehensive calculus operations including derivatives, integrals, limits, and series*")
    
    # Operation selection tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Derivatives", "Integrals", "Limits", "Series", "Function Analysis", "Multivariable Calculus", "Differential Equations"
    ])
    
    with tab1:
        render_derivatives_section()
    
    with tab2:
        render_integrals_section()
    
    with tab3:
        render_limits_section()
    
    with tab4:
        render_series_section()
    
    with tab5:
        render_function_analysis_section()
    
    with tab6:
        render_multivariable_calculus_section()
    
    with tab7:
        render_differential_equations_section()

def render_derivatives_section():
    """Render derivatives calculation section"""
    st.subheader("Derivative Calculator")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x):",
            placeholder="x^3 + 2*x^2 - 3*x + 1",
            key="derivative_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value="x", key="derivative_var")
    
    with col3:
        order = st.number_input("Order:", min_value=1, max_value=5, value=1, key="derivative_order")
    
    if expression:
        # Single variable derivative
        st.markdown("**Single Variable Derivative:**")
        result = calculus_engine.compute_derivative(expression, variable, order)
        display_derivative_result(result, order)
        
        # Note: Partial derivatives have been moved to the Multivariable Calculus tab

def render_integrals_section():
    """Render integrals calculation section"""
    st.subheader("Integral Calculator")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x):",
            placeholder="x^2 + sin(x)",
            key="integral_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value="x", key="integral_var")
    
    # Integral type selection
    integral_type = st.radio(
        "Integral Type:",
        ["Indefinite", "Definite"],
        key="integral_type"
    )
    
    # Initialize limits with default values
    lower_limit = 0.0
    upper_limit = 1.0
    
    # Definite integral limits
    if integral_type == "Definite":
        col1, col2 = st.columns(2)
        with col1:
            lower_limit = st.number_input("Lower Limit:", value=0.0, key="lower_limit")
        with col2:
            upper_limit = st.number_input("Upper Limit:", value=1.0, key="upper_limit")
    
    if expression:
        result = None
        if integral_type == "Indefinite":
            result = calculus_engine.compute_integral(expression, variable, definite=False)
        elif integral_type == "Definite":
            result = calculus_engine.compute_integral(
                expression, variable, definite=True, 
                lower_limit=lower_limit, upper_limit=upper_limit
            )
        
        if result:
            display_integral_result(result, integral_type)
            
            # Numerical integration comparison for definite integrals
            if integral_type == "Definite" and result['success']:
                st.markdown("---")
                st.markdown("**Numerical Integration Verification:**")
                perform_numerical_integration(expression, variable, lower_limit, upper_limit)

def render_limits_section():
    """Render limits calculation section"""
    st.subheader("Limit Calculator")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x):",
            placeholder="sin(x)/x",
            key="limit_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value="x", key="limit_var")
    
    col1, col2 = st.columns(2)
    
    with col1:
        limit_point = st.text_input(
            "Limit Point:",
            value="0",
            placeholder="Enter number or 'oo' for infinity",
            key="limit_point"
        )
    
    with col2:
        direction = st.selectbox(
            "Direction:",
            ["+-", "+", "-"],
            help="+-: both sides, +: from right, -: from left",
            key="limit_direction"
        )
    
    if expression and limit_point:
        # Convert limit point
        try:
            if limit_point.lower() in ['oo', 'inf', 'infinity']:
                limit_val = 'oo'
            elif limit_point == '-oo':
                limit_val = '-oo'
            else:
                limit_val = float(limit_point)
        except ValueError:
            limit_val = limit_point
        
        result = calculus_engine.compute_limit(expression, variable, limit_val, direction)
        display_limit_result(result, limit_point, direction)

def render_series_section():
    """Render series expansion section"""
    st.subheader("Series Expansion")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x):",
            placeholder="exp(x)",
            key="series_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value="x", key="series_var")
    
    col1, col2 = st.columns(2)
    
    with col1:
        point = st.number_input("Expansion Point:", value=0.0, key="series_point")
    
    with col2:
        order = st.number_input("Order:", min_value=1, max_value=20, value=6, key="series_order")
    
    if expression:
        result = calculus_engine.compute_series(expression, variable, point, order)
        display_series_result(result, point, order)
        
        # Series convergence visualization
        if result['success']:
            st.markdown("---")
            st.markdown("**Series Convergence Visualization:**")
            visualize_series_convergence(expression, variable, point, order)

def render_function_analysis_section():
    """Render comprehensive function analysis section"""
    st.subheader("Function Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x):",
            placeholder="x^3 - 3*x^2 + 2*x",
            key="analysis_expr"
        )
    
    with col2:
        variable = st.text_input("Variable:", value="x", key="analysis_var")
    
    if expression:
        result = calculus_engine.analyze_function(expression, variable)
        display_function_analysis(result)
        
        # Critical points analysis
        if result['success'] and result['critical_points']:
            st.markdown("---")
            st.markdown("**Critical Points Analysis:**")
            analyze_critical_points(expression, variable, result['critical_points'])

def display_derivative_result(result, order):
    """Display derivative calculation results"""
    if result['success']:
        st.success(f"✅ {order}{'st' if order == 1 else 'nd' if order == 2 else 'rd' if order == 3 else 'th'} derivative calculated successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Function:**")
            st.latex(sp.latex(result['original_expression']))
        
        with col2:
            st.markdown(f"**{order}{'st' if order == 1 else 'nd' if order == 2 else 'rd' if order == 3 else 'th'} Derivative:**")
            st.latex(sp.latex(result['simplified_derivative']))
        
        # Step-by-step if available
        if str(result['derivative']) != str(result['simplified_derivative']):
            st.markdown("**Step-by-step:**")
            st.write("1. Raw derivative:")
            st.latex(sp.latex(result['derivative']))
            st.write("2. Simplified:")
            st.latex(sp.latex(result['simplified_derivative']))
    
    else:
        st.error(f"❌ Derivative calculation failed: {result['error']}")

def display_integral_result(result, integral_type):
    """Display integral calculation results"""
    if result['success']:
        st.success(f"✅ {integral_type} integral calculated successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Function:**")
            st.latex(sp.latex(result['original_expression']))
        
        with col2:
            st.markdown(f"**{integral_type} Integral:**")
            st.latex(sp.latex(result['simplified_integral']))
            
            if result['numeric_value'] is not None:
                st.metric("Numeric Value", f"{result['numeric_value']:.10g}")
        
        # Step-by-step if available
        if str(result['integral']) != str(result['simplified_integral']):
            st.markdown("**Step-by-step:**")
            st.write("1. Raw integral:")
            st.latex(sp.latex(result['integral']))
            st.write("2. Simplified:")
            st.latex(sp.latex(result['simplified_integral']))
    
    else:
        st.error(f"❌ Integral calculation failed: {result['error']}")

def display_limit_result(result, limit_point, direction):
    """Display limit calculation results"""
    if result['success']:
        st.success("✅ Limit calculated successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Function:**")
            st.latex(sp.latex(result['original_expression']))
        
        with col2:
            st.markdown("**Limit Result:**")
            limit_latex = f"\\lim_{{x \\to {limit_point}{'⁺' if direction == '+' else '⁻' if direction == '-' else ''}}} f(x) = {sp.latex(result['limit'])}"
            st.latex(limit_latex)
        
        # Interpret result
        if result['limit'] == sp.oo:
            st.info("🔼 Limit approaches positive infinity")
        elif result['limit'] == -sp.oo:
            st.info("🔽 Limit approaches negative infinity")
        elif result['limit'].has(sp.oo):
            st.warning("⚠️ Limit does not exist (involves infinity)")
        else:
            st.info(f"➡️ Limit exists and equals {result['limit']}")
    
    else:
        st.error(f"❌ Limit calculation failed: {result['error']}")

def display_series_result(result, point, order):
    """Display series expansion results"""
    if result['success']:
        st.success("✅ Series expansion calculated successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Function:**")
            st.latex(sp.latex(result['original_expression']))
        
        with col2:
            series_type = "Maclaurin" if point == 0 else "Taylor"
            st.markdown(f"**{series_type} Series (order {order}):**")
            st.latex(sp.latex(result['series']))
        
        # Series information
        st.info(f"ℹ️ {series_type} series expansion around x = {point} up to order {order}")
    
    else:
        st.error(f"❌ Series expansion failed: {result['error']}")

def display_function_analysis(result):
    """Display comprehensive function analysis"""
    if result['success']:
        st.success("✅ Function analysis completed successfully")
        
        # Derivatives
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**First Derivative:**")
            st.latex(sp.latex(result['first_derivative']))
        
        with col2:
            st.markdown("**Second Derivative:**")
            st.latex(sp.latex(result['second_derivative']))
        
        # Critical points and inflection points
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Critical Points:**")
            if result['critical_points']:
                for i, point in enumerate(result['critical_points']):
                    st.write(f"x_{i+1} = {point}")
            else:
                st.write("No critical points found")
        
        with col2:
            st.markdown("**Inflection Points:**")
            if result['inflection_points']:
                for i, point in enumerate(result['inflection_points']):
                    st.write(f"x_{i+1} = {point}")
            else:
                st.write("No inflection points found")
    
    else:
        st.error(f"❌ Function analysis failed: {result['error']}")

def compute_partial_derivatives(expression, variables):
    """Compute and display partial derivatives"""
    for var in variables:
        result = calculus_engine.compute_partial_derivative(expression, var)
        if result['success']:
            st.write(f"**∂f/∂{var}:**")
            st.latex(sp.latex(result['simplified_partial']))
        else:
            st.error(f"Error computing ∂f/∂{var}: {result['error']}")

def display_gradient_result(result):
    """Display gradient calculation results"""
    if result['success']:
        st.success("✅ Gradient calculated successfully")
        
        st.markdown("**Gradient Vector:**")
        gradient_components = []
        for var, partial in result['gradient'].items():
            gradient_components.append(f"\\frac{{\\partial f}}{{\\partial {var}}} = {sp.latex(partial)}")
        
        gradient_latex = "\\nabla f = \\begin{pmatrix} " + " \\\\ ".join(gradient_components) + " \\end{pmatrix}"
        st.latex(gradient_latex)
    
    else:
        st.error(f"❌ Gradient calculation failed: {result['error']}")

def perform_numerical_integration(expression, variable, lower_limit, upper_limit):
    """Perform numerical integration for comparison"""
    try:
        from scipy import integrate
        
        # Create numerical function
        expr = expression_parser.parse(expression)
        var_symbol = sp.Symbol(variable)
        func = sp.lambdify(var_symbol, expr, 'numpy')
        
        # Numerical integration
        numerical_result, error = integrate.quad(func, lower_limit, upper_limit)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Numerical Result", f"{numerical_result:.10g}")
        
        with col2:
            st.metric("Estimated Error", f"{error:.2e}")
    
    except Exception as e:
        st.error(f"Numerical integration failed: {str(e)}")

def visualize_series_convergence(expression, variable, point, max_order):
    """Visualize series convergence"""
    try:
        # Create plot data
        x_vals = np.linspace(point - 2, point + 2, 1000)
        
        # Original function
        expr = expression_parser.parse(expression)
        var_symbol = sp.Symbol(variable)
        original_func = sp.lambdify(var_symbol, expr, 'numpy')
        
        try:
            y_original = original_func(x_vals)
        except:
            st.warning("Cannot plot original function (may have singularities)")
            return
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Original function
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_original,
            mode='lines',
            name='Original Function',
            line=dict(color='black', width=3)
        ))
        
        # Series approximations
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for order in [1, 2, 3, 4, min(max_order, 6)]:
            series_result = calculus_engine.compute_series(expression, variable, point, order)
            if series_result['success']:
                series_func = sp.lambdify(var_symbol, series_result['series'], 'numpy')
                try:
                    y_series = series_func(x_vals)
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_series,
                        mode='lines',
                        name=f'Order {order}',
                        line=dict(color=colors[order-1], width=2, dash='dash')
                    ))
                except:
                    continue
        
        fig.update_layout(
            title="Series Convergence Visualization",
            xaxis_title=variable,
            yaxis_title="f(x)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Visualization failed: {str(e)}")

def analyze_critical_points(expression, variable, critical_points):
    """Analyze the nature of critical points"""
    try:
        # Second derivative test
        second_derivative_result = calculus_engine.compute_derivative(expression, variable, 2)
        
        if second_derivative_result['success']:
            second_deriv = second_derivative_result['simplified_derivative']
            var_symbol = sp.Symbol(variable)
            
            for i, point_str in enumerate(critical_points):
                try:
                    point = float(point_str)
                    second_deriv_value = float(second_deriv.subs(var_symbol, point))
                    
                    if second_deriv_value > 0:
                        nature = "Local Minimum"
                        icon = "🔻"
                    elif second_deriv_value < 0:
                        nature = "Local Maximum"
                        icon = "🔺"
                    else:
                        nature = "Inconclusive (may be inflection point)"
                        icon = "❓"
                    
                    st.write(f"{icon} **x = {point}**: {nature}")
                    st.write(f"   Second derivative value: {second_deriv_value:.6g}")
                
                except (ValueError, TypeError):
                    st.write(f"❓ **x = {point_str}**: Could not analyze (complex or symbolic)")
    
    except Exception as e:
        st.error(f"Critical point analysis failed: {str(e)}")

def render_multivariable_calculus_section():
    """Render multivariable calculus section"""
    st.subheader("Multivariable Calculus")
    st.markdown("*Partial derivatives, gradients, multiple integrals, and vector calculus*")
    
    # Sub-tabs for different multivariable operations
    subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs([
        "Partial Derivatives", "Gradients & Directional", "Multiple Integrals", "Vector Fields", "Optimization"
    ])
    
    with subtab1:
        render_partial_derivatives()
    
    with subtab2:
        render_gradients_section()
    
    with subtab3:
        render_multiple_integrals()
    
    with subtab4:
        render_vector_fields()
    
    with subtab5:
        render_optimization_section()

def render_partial_derivatives():
    """Render partial derivatives section"""
    st.markdown("**Partial Derivatives**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x,y,z,...):",
            placeholder="x**2*y + y**2*z + sin(x*y)",
            key="partial_expr"
        )
    
    with col2:
        variables = st.text_input(
            "Variables (comma-separated):",
            value="x,y,z",
            key="partial_vars_unique"
        )
    
    if expression and variables:
        var_list = [v.strip() for v in variables.split(',') if v.strip()]
        
        if st.button("Compute Partial Derivatives", key="compute_partials"):
            try:
                expr = expression_parser.parse(expression)
                
                st.markdown("**Partial Derivatives:**")
                
                for var in var_list:
                    try:
                        result = calculus_engine.compute_partial_derivative(expression, var)
                        if result['success']:
                            st.markdown(f"∂f/∂{var} = `{result['partial_derivative']}`")
                        else:
                            st.error(f"Failed to compute ∂f/∂{var}: {result['error']}")
                    except Exception as e:
                        st.error(f"Error computing ∂f/∂{var}: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error parsing expression: {str(e)}")
    
    # Second order partial derivatives
    st.markdown("---")
    st.markdown("**Second-Order Partial Derivatives**")
    
    if st.checkbox("Compute second-order partials", key="second_order_check"):
        if expression and variables:
            var_list = [v.strip() for v in variables.split(',') if v.strip()]
            
            if st.button("Compute Second-Order Partials", key="compute_second_partials"):
                try:
                    expr = expression_parser.parse(expression)
                    
                    for i, var1 in enumerate(var_list):
                        for j, var2 in enumerate(var_list):
                            if i <= j:  # Avoid redundant calculations due to symmetry
                                try:
                                    # First compute ∂f/∂var1
                                    first_deriv = sp.diff(expr, sp.Symbol(var1))
                                    # Then compute ∂²f/∂var2∂var1
                                    second_deriv = sp.diff(first_deriv, sp.Symbol(var2))
                                    
                                    if var1 == var2:
                                        st.markdown(f"∂²f/∂{var1}² = `{second_deriv}`")
                                    else:
                                        st.markdown(f"∂²f/∂{var2}∂{var1} = `{second_deriv}`")
                                        
                                except Exception as e:
                                    st.error(f"Error computing ∂²f/∂{var2}∂{var1}: {str(e)}")
                                    
                except Exception as e:
                    st.error(f"Error in second-order computation: {str(e)}")

def render_gradients_section():
    """Render gradients and directional derivatives section"""
    st.markdown("**Gradient Vector**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x,y,z,...):",
            placeholder="x**2 + y**2 + z**2",
            key="gradient_expr"
        )
    
    with col2:
        variables = st.text_input(
            "Variables:",
            value="x,y,z",
            key="gradient_vars_unique"
        )
    
    if expression and variables:
        var_list = [v.strip() for v in variables.split(',') if v.strip()]
        
        if st.button("Compute Gradient", key="compute_gradient"):
            try:
                result = calculus_engine.compute_gradient(expression, var_list)
                
                if result['success']:
                    st.markdown("**Gradient Vector:**")
                    gradient_components = result['gradient']
                    
                    gradient_str = "∇f = ("
                    gradient_str += ", ".join([f"`{comp}`" for comp in gradient_components])
                    gradient_str += ")"
                    
                    st.markdown(gradient_str)
                    
                    # Display magnitude
                    if 'magnitude' in result:
                        st.markdown(f"**Magnitude:** `{result['magnitude']}`")
                else:
                    st.error(f"Gradient computation failed: {result['error']}")
                    
            except Exception as e:
                st.error(f"Error computing gradient: {str(e)}")
    
    # Directional derivatives
    st.markdown("---")
    st.markdown("**Directional Derivative**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        point_coords = st.text_input(
            "Point coordinates (comma-separated):",
            placeholder="1,2,3",
            key="direction_point"
        )
    
    with col2:
        direction_vector = st.text_input(
            "Direction vector (comma-separated):",
            placeholder="1,1,1",
            key="direction_vec"
        )
    
    if expression and variables and point_coords and direction_vector:
        if st.button("Compute Directional Derivative", key="compute_directional"):
            try:
                var_list = [v.strip() for v in variables.split(',') if v.strip()]
                point_vals = [float(v.strip()) for v in point_coords.split(',')]
                dir_vals = [float(v.strip()) for v in direction_vector.split(',')]
                
                if len(point_vals) != len(var_list) or len(dir_vals) != len(var_list):
                    st.error("Number of coordinates and direction components must match number of variables")
                else:
                    # Compute gradient
                    gradient_result = calculus_engine.compute_gradient(expression, var_list)
                    
                    if gradient_result['success']:
                        gradient = gradient_result['gradient']
                        
                        # Evaluate gradient at the point
                        grad_at_point = []
                        for i, grad_comp in enumerate(gradient):
                            subs_dict = {sp.Symbol(var_list[j]): point_vals[j] for j in range(len(var_list))}
                            grad_val = float(grad_comp.subs(subs_dict))
                            grad_at_point.append(grad_val)
                        
                        # Normalize direction vector
                        dir_magnitude = sum(x**2 for x in dir_vals)**0.5
                        unit_dir = [x/dir_magnitude for x in dir_vals]
                        
                        # Compute directional derivative
                        dir_deriv = sum(grad_at_point[i] * unit_dir[i] for i in range(len(grad_at_point)))
                        
                        st.markdown(f"**Directional Derivative:** `{dir_deriv:.6f}`")
                        st.markdown(f"**Gradient at point:** `{grad_at_point}`")
                        st.markdown(f"**Unit direction vector:** `{unit_dir}`")
                    else:
                        st.error(f"Failed to compute gradient: {gradient_result['error']}")
                        
            except Exception as e:
                st.error(f"Error computing directional derivative: {str(e)}")

def render_multiple_integrals():
    """Render multiple integrals section"""
    st.markdown("**Multiple Integrals**")
    st.info("Double and triple integrals with specified bounds")
    
    # Integral type selection
    integral_type = st.selectbox(
        "Integral Type:",
        ["Double Integral", "Triple Integral"],
        key="multiple_integral_type"
    )
    
    if integral_type == "Double Integral":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            expression = st.text_input(
                "Integrand f(x,y):",
                placeholder="x*y + x**2",
                key="double_integral_expr"
            )
        
        with col2:
            order = st.selectbox(
                "Integration Order:",
                ["dx dy", "dy dx"],
                key="double_order"
            )
        
        # Bounds input
        st.markdown("**Integration Bounds:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_lower = st.text_input("x lower:", value="0", key="x_lower_double")
        with col2:
            x_upper = st.text_input("x upper:", value="1", key="x_upper_double")
        with col3:
            y_lower = st.text_input("y lower:", value="0", key="y_lower_double")
        with col4:
            y_upper = st.text_input("y upper:", value="1", key="y_upper_double")
        
        if st.button("Compute Double Integral", key="compute_double"):
            try:
                expr = expression_parser.parse(expression)
                
                # Parse bounds
                x_l = expression_parser.parse(x_lower) if x_lower else 0
                x_u = expression_parser.parse(x_upper) if x_upper else 1
                y_l = expression_parser.parse(y_lower) if y_lower else 0
                y_u = expression_parser.parse(y_upper) if y_upper else 1
                
                # Compute integral based on order
                if order == "dx dy":
                    # First integrate with respect to x, then y
                    inner = sp.integrate(expr, (sp.Symbol('x'), x_l, x_u))
                    result = sp.integrate(inner, (sp.Symbol('y'), y_l, y_u))
                else:
                    # First integrate with respect to y, then x
                    inner = sp.integrate(expr, (sp.Symbol('y'), y_l, y_u))
                    result = sp.integrate(inner, (sp.Symbol('x'), x_l, x_u))
                
                st.markdown(f"**Result:** `{result}`")
                
                # Try to evaluate numerically
                try:
                    numeric_result = float(result)
                    st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
                except:
                    st.markdown("*Result contains symbolic expressions*")
                    
            except Exception as e:
                st.error(f"Error computing double integral: {str(e)}")
    
    else:  # Triple Integral
        col1, col2 = st.columns([3, 1])
        
        with col1:
            expression = st.text_input(
                "Integrand f(x,y,z):",
                placeholder="x*y*z",
                key="triple_integral_expr"
            )
        
        with col2:
            order = st.selectbox(
                "Integration Order:",
                ["dx dy dz", "dx dz dy", "dy dx dz", "dy dz dx", "dz dx dy", "dz dy dx"],
                key="triple_order"
            )
        
        # Bounds input
        st.markdown("**Integration Bounds:**")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            x_lower = st.text_input("x lower:", value="0", key="x_lower_triple")
        with col2:
            x_upper = st.text_input("x upper:", value="1", key="x_upper_triple")
        with col3:
            y_lower = st.text_input("y lower:", value="0", key="y_lower_triple")
        with col4:
            y_upper = st.text_input("y upper:", value="1", key="y_upper_triple")
        with col5:
            z_lower = st.text_input("z lower:", value="0", key="z_lower_triple")
        with col6:
            z_upper = st.text_input("z upper:", value="1", key="z_upper_triple")
        
        if st.button("Compute Triple Integral", key="compute_triple"):
            try:
                expr = expression_parser.parse(expression)
                
                # Parse bounds
                x_l = expression_parser.parse(x_lower) if x_lower else 0
                x_u = expression_parser.parse(x_upper) if x_upper else 1
                y_l = expression_parser.parse(y_lower) if y_lower else 0
                y_u = expression_parser.parse(y_upper) if y_upper else 1
                z_l = expression_parser.parse(z_lower) if z_lower else 0
                z_u = expression_parser.parse(z_upper) if z_upper else 1
                
                # Compute triple integral based on order
                order_mapping = {
                    "dx dy dz": [(sp.Symbol('x'), x_l, x_u), (sp.Symbol('y'), y_l, y_u), (sp.Symbol('z'), z_l, z_u)],
                    "dx dz dy": [(sp.Symbol('x'), x_l, x_u), (sp.Symbol('z'), z_l, z_u), (sp.Symbol('y'), y_l, y_u)],
                    "dy dx dz": [(sp.Symbol('y'), y_l, y_u), (sp.Symbol('x'), x_l, x_u), (sp.Symbol('z'), z_l, z_u)],
                    "dy dz dx": [(sp.Symbol('y'), y_l, y_u), (sp.Symbol('z'), z_l, z_u), (sp.Symbol('x'), x_l, x_u)],
                    "dz dx dy": [(sp.Symbol('z'), z_l, z_u), (sp.Symbol('x'), x_l, x_u), (sp.Symbol('y'), y_l, y_u)],
                    "dz dy dx": [(sp.Symbol('z'), z_l, z_u), (sp.Symbol('y'), y_l, y_u), (sp.Symbol('x'), x_l, x_u)]
                }
                
                integration_order = order_mapping[order]
                
                result = expr
                for var, lower, upper in integration_order:
                    result = sp.integrate(result, (var, lower, upper))
                
                st.markdown(f"**Result:** `{result}`")
                
                # Try to evaluate numerically
                try:
                    numeric_result = float(result)
                    st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
                except:
                    st.markdown("*Result contains symbolic expressions*")
                    
            except Exception as e:
                st.error(f"Error computing triple integral: {str(e)}")

def render_vector_fields():
    """Render comprehensive vector calculus section"""
    st.markdown("**Vector Calculus**")
    st.markdown("*Comprehensive vector field analysis, line integrals, surface integrals, and flux calculations*")
    
    # Sub-tabs for vector calculus
    vector_tab1, vector_tab2, vector_tab3, vector_tab4 = st.tabs([
        "Vector Fields", "Line Integrals", "Surface Integrals", "Theorems"
    ])
    
    with vector_tab1:
        render_vector_field_analysis()
    
    with vector_tab2:
        render_line_integrals()
    
    with vector_tab3:
        render_surface_integrals()
    
    with vector_tab4:
        render_vector_theorems()

def render_vector_field_analysis():
    """Vector field analysis (divergence, curl, etc.)"""
    st.subheader("Vector Field Analysis")
    
    # Vector field input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        F_x = st.text_input(
            "F_x component:",
            placeholder="y",
            key="vector_fx"
        )
    
    with col2:
        F_y = st.text_input(
            "F_y component:",
            placeholder="-x",
            key="vector_fy"
        )
    
    with col3:
        F_z = st.text_input(
            "F_z component (optional):",
            placeholder="z",
            key="vector_fz"
        )
    
    if F_x and F_y:
        # Divergence and curl calculations
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Compute Divergence", key="compute_divergence"):
                try:
                    fx_expr = expression_parser.parse(F_x)
                    fy_expr = expression_parser.parse(F_y)
                    
                    div_x = sp.diff(fx_expr, sp.Symbol('x'))
                    div_y = sp.diff(fy_expr, sp.Symbol('y'))
                    
                    if F_z:
                        fz_expr = expression_parser.parse(F_z)
                        div_z = sp.diff(fz_expr, sp.Symbol('z'))
                        divergence = div_x + div_y + div_z
                    else:
                        divergence = div_x + div_y
                    
                    st.markdown(f"**Divergence:** `{divergence}`")
                    
                    # Physical interpretation
                    if divergence == 0:
                        st.info("🌊 **Incompressible flow** - No sources or sinks")
                    else:
                        st.info("📈 **Compressible flow** - Has sources/sinks")
                    
                except Exception as e:
                    st.error(f"Error computing divergence: {str(e)}")
        
        with col2:
            if st.button("Compute Curl", key="compute_curl"):
                try:
                    fx_expr = expression_parser.parse(F_x)
                    fy_expr = expression_parser.parse(F_y)
                    
                    if F_z:
                        fz_expr = expression_parser.parse(F_z)
                        
                        curl_x = sp.diff(fz_expr, sp.Symbol('y')) - sp.diff(fy_expr, sp.Symbol('z'))
                        curl_y = sp.diff(fx_expr, sp.Symbol('z')) - sp.diff(fz_expr, sp.Symbol('x'))
                        curl_z = sp.diff(fy_expr, sp.Symbol('x')) - sp.diff(fx_expr, sp.Symbol('y'))
                        
                        st.markdown("**Curl (3D):**")
                        st.markdown(f"curl_x = `{curl_x}`")
                        st.markdown(f"curl_y = `{curl_y}`")
                        st.markdown(f"curl_z = `{curl_z}`")
                        
                        curl_magnitude = sp.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
                        st.markdown(f"**Curl Magnitude:** `{curl_magnitude}`")
                    else:
                        curl = sp.diff(fy_expr, sp.Symbol('x')) - sp.diff(fx_expr, sp.Symbol('y'))
                        st.markdown(f"**Curl (2D):** `{curl}`")
                        
                        if curl == 0:
                            st.info("🔄 **Conservative field** - No rotation")
                        else:
                            st.info("🌪️ **Non-conservative field** - Has rotation")
                    
                except Exception as e:
                    st.error(f"Error computing curl: {str(e)}")

def render_line_integrals():
    """Line integrals of scalar and vector fields"""
    st.subheader("Line Integrals")
    
    integral_type = st.selectbox(
        "Line Integral Type:",
        ["Scalar Field", "Vector Field", "Arc Length"],
        key="line_integral_type"
    )
    
    if integral_type == "Scalar Field":
        st.markdown("**Line Integral of Scalar Field:** ∫_C f(x,y,z) ds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scalar_field = st.text_input(
                "Scalar field f(x,y,z):",
                placeholder="x**2 + y**2",
                key="scalar_field_line"
            )
        
        with col2:
            parametric_path = st.text_area(
                "Parametric path (one per line):\nx(t) =\ny(t) =\nz(t) = (optional)",
                value="cos(t)\nsin(t)\n",
                key="parametric_path_scalar"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            t_start = st.number_input("t start:", value=0.0, key="t_start_scalar")
        with col2:
            t_end = st.number_input("t end:", value=6.28, key="t_end_scalar")
        
        if st.button("Compute Scalar Line Integral", key="compute_scalar_line"):
            compute_scalar_line_integral(scalar_field, parametric_path, t_start, t_end)
    
    elif integral_type == "Vector Field":
        st.markdown("**Line Integral of Vector Field:** ∫_C F⃗·dr⃗")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            F_x = st.text_input("F_x:", placeholder="y", key="vector_line_fx")
        with col2:
            F_y = st.text_input("F_y:", placeholder="-x", key="vector_line_fy")
        with col3:
            F_z = st.text_input("F_z:", placeholder="0", key="vector_line_fz")
        
        parametric_path = st.text_area(
            "Parametric path (one per line):\nx(t) =\ny(t) =\nz(t) = (optional)",
            value="cos(t)\nsin(t)\n0",
            key="parametric_path_vector"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            t_start = st.number_input("t start:", value=0.0, key="t_start_vector")
        with col2:
            t_end = st.number_input("t end:", value=6.28, key="t_end_vector")
        
        if st.button("Compute Vector Line Integral", key="compute_vector_line"):
            compute_vector_line_integral(F_x, F_y, F_z, parametric_path, t_start, t_end)

def render_surface_integrals():
    """Surface integrals of scalar and vector fields"""
    st.subheader("Surface Integrals")
    
    integral_type = st.selectbox(
        "Surface Integral Type:",
        ["Scalar Field", "Vector Field (Flux)", "Surface Area"],
        key="surface_integral_type"
    )
    
    if integral_type == "Scalar Field":
        st.markdown("**Surface Integral of Scalar Field:** ∬_S f(x,y,z) dS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scalar_field = st.text_input(
                "Scalar field f(x,y,z):",
                placeholder="x**2 + y**2 + z**2",
                key="scalar_field_surface"
            )
        
        with col2:
            surface_equation = st.text_input(
                "Surface z = g(x,y):",
                placeholder="x**2 + y**2",
                key="surface_equation"
            )
        
        # Domain bounds
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_min = st.number_input("x min:", value=-1.0, key="x_min_surface")
        with col2:
            x_max = st.number_input("x max:", value=1.0, key="x_max_surface")
        with col3:
            y_min = st.number_input("y min:", value=-1.0, key="y_min_surface")
        with col4:
            y_max = st.number_input("y max:", value=1.0, key="y_max_surface")
        
        if st.button("Compute Scalar Surface Integral", key="compute_scalar_surface"):
            compute_scalar_surface_integral(scalar_field, surface_equation, x_min, x_max, y_min, y_max)
    
    elif integral_type == "Vector Field (Flux)":
        st.markdown("**Flux through Surface:** ∬_S F⃗·n̂ dS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            F_x = st.text_input("F_x:", placeholder="x", key="flux_fx")
        with col2:
            F_y = st.text_input("F_y:", placeholder="y", key="flux_fy")
        with col3:
            F_z = st.text_input("F_z:", placeholder="z", key="flux_fz")
        
        surface_equation = st.text_input(
            "Surface z = g(x,y):",
            placeholder="1 - x**2 - y**2",
            key="flux_surface"
        )
        
        # Domain bounds
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            x_min = st.number_input("x min:", value=-1.0, key="x_min_flux")
        with col2:
            x_max = st.number_input("x max:", value=1.0, key="x_max_flux")
        with col3:
            y_min = st.number_input("y min:", value=-1.0, key="y_min_flux")
        with col4:
            y_max = st.number_input("y max:", value=1.0, key="y_max_flux")
        
        if st.button("Compute Flux Integral", key="compute_flux"):
            compute_flux_integral(F_x, F_y, F_z, surface_equation, x_min, x_max, y_min, y_max)

def render_vector_theorems():
    """Vector calculus theorems (Green's, Stokes', Divergence)"""
    st.subheader("Vector Calculus Theorems")
    
    theorem = st.selectbox(
        "Select Theorem:",
        ["Green's Theorem", "Stokes' Theorem", "Divergence Theorem"],
        key="vector_theorem"
    )
    
    if theorem == "Green's Theorem":
        st.markdown("**Green's Theorem:** ∮_C (P dx + Q dy) = ∬_D (∂Q/∂x - ∂P/∂y) dA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            P_expr = st.text_input("P(x,y):", placeholder="y", key="green_P")
        with col2:
            Q_expr = st.text_input("Q(x,y):", placeholder="-x", key="green_Q")
        
        st.info("**Verification:** Compute both line integral around boundary and double integral over region")
        
        if P_expr and Q_expr:
            if st.button("Verify Green's Theorem", key="verify_green"):
                verify_greens_theorem(P_expr, Q_expr)
    
    elif theorem == "Stokes' Theorem":
        st.markdown("**Stokes' Theorem:** ∮_C F⃗·dr⃗ = ∬_S (∇×F⃗)·n̂ dS")
        st.info("**Relates:** Line integral around boundary to surface integral of curl")
        
    elif theorem == "Divergence Theorem":
        st.markdown("**Divergence Theorem:** ∬_S F⃗·n̂ dS = ∭_V ∇·F⃗ dV")
        st.info("**Relates:** Flux through closed surface to volume integral of divergence")

def compute_scalar_line_integral(scalar_field, parametric_path, t_start, t_end):
    """Compute line integral of scalar field"""
    try:
        # Parse scalar field
        f_expr = expression_parser.parse(scalar_field)
        
        # Parse parametric path
        path_lines = parametric_path.strip().split('\n')
        x_param = expression_parser.parse(path_lines[0]) if len(path_lines) > 0 else sp.Symbol('t')
        y_param = expression_parser.parse(path_lines[1]) if len(path_lines) > 1 else sp.Symbol('t')
        z_param = expression_parser.parse(path_lines[2]) if len(path_lines) > 2 else 0
        
        t = sp.Symbol('t')
        
        # Substitute parametric equations into scalar field
        f_parametric = f_expr.subs([(sp.Symbol('x'), x_param), (sp.Symbol('y'), y_param), (sp.Symbol('z'), z_param)])
        
        # Compute derivatives for arc length element
        dx_dt = sp.diff(x_param, t)
        dy_dt = sp.diff(y_param, t)
        dz_dt = sp.diff(z_param, t)
        
        # Arc length element ds = √((dx/dt)² + (dy/dt)² + (dz/dt)²) dt
        ds_dt = sp.sqrt(dx_dt**2 + dy_dt**2 + dz_dt**2)
        
        # Integrand
        integrand = f_parametric * ds_dt
        
        # Compute integral
        result = sp.integrate(integrand, (t, t_start, t_end))
        
        st.markdown(f"**Arc length element:** `ds = {ds_dt} dt`")
        st.markdown(f"**Integrand:** `{integrand}`")
        st.markdown(f"**Result:** `{result}`")
        
        # Try numerical evaluation
        try:
            numeric_result = float(result)
            st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
        except:
            st.markdown("*Result contains symbolic expressions*")
            
    except Exception as e:
        st.error(f"Error computing scalar line integral: {str(e)}")

def compute_vector_line_integral(F_x, F_y, F_z, parametric_path, t_start, t_end):
    """Compute line integral of vector field"""
    try:
        # Parse vector field components
        fx_expr = expression_parser.parse(F_x) if F_x else 0
        fy_expr = expression_parser.parse(F_y) if F_y else 0
        fz_expr = expression_parser.parse(F_z) if F_z else 0
        
        # Parse parametric path
        path_lines = parametric_path.strip().split('\n')
        x_param = expression_parser.parse(path_lines[0]) if len(path_lines) > 0 else sp.Symbol('t')
        y_param = expression_parser.parse(path_lines[1]) if len(path_lines) > 1 else sp.Symbol('t')
        z_param = expression_parser.parse(path_lines[2]) if len(path_lines) > 2 else 0
        
        t = sp.Symbol('t')
        
        # Substitute parametric equations into vector field
        fx_parametric = fx_expr.subs([(sp.Symbol('x'), x_param), (sp.Symbol('y'), y_param), (sp.Symbol('z'), z_param)])
        fy_parametric = fy_expr.subs([(sp.Symbol('x'), x_param), (sp.Symbol('y'), y_param), (sp.Symbol('z'), z_param)])
        fz_parametric = fz_expr.subs([(sp.Symbol('x'), x_param), (sp.Symbol('y'), y_param), (sp.Symbol('z'), z_param)])
        
        # Compute dr/dt
        dx_dt = sp.diff(x_param, t)
        dy_dt = sp.diff(y_param, t)
        dz_dt = sp.diff(z_param, t)
        
        # Dot product F⃗·dr⃗/dt
        integrand = fx_parametric * dx_dt + fy_parametric * dy_dt + fz_parametric * dz_dt
        
        # Compute integral
        result = sp.integrate(integrand, (t, t_start, t_end))
        
        st.markdown(f"**dr/dt:** `({dx_dt}, {dy_dt}, {dz_dt})`")
        st.markdown(f"**F⃗·dr/dt:** `{integrand}`")
        st.markdown(f"**Result:** `{result}`")
        
        # Try numerical evaluation
        try:
            numeric_result = float(result)
            st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
        except:
            st.markdown("*Result contains symbolic expressions*")
            
    except Exception as e:
        st.error(f"Error computing vector line integral: {str(e)}")

def compute_scalar_surface_integral(scalar_field, surface_equation, x_min, x_max, y_min, y_max):
    """Compute surface integral of scalar field"""
    try:
        # Parse expressions
        f_expr = expression_parser.parse(scalar_field)
        z_expr = expression_parser.parse(surface_equation)
        
        x, y, z = sp.symbols('x y z')
        
        # Substitute surface equation into scalar field
        f_surface = f_expr.subs(z, z_expr)
        
        # Compute partial derivatives for surface element
        dz_dx = sp.diff(z_expr, x)
        dz_dy = sp.diff(z_expr, y)
        
        # Surface element dS = √(1 + (∂z/∂x)² + (∂z/∂y)²) dx dy
        dS = sp.sqrt(1 + dz_dx**2 + dz_dy**2)
        
        # Integrand
        integrand = f_surface * dS
        
        # Compute double integral
        result = sp.integrate(integrand, (x, x_min, x_max), (y, y_min, y_max))
        
        st.markdown(f"**Surface element:** `dS = {dS} dx dy`")
        st.markdown(f"**Integrand:** `{integrand}`")
        st.markdown(f"**Result:** `{result}`")
        
        # Try numerical evaluation
        try:
            numeric_result = float(result)
            st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
        except:
            st.markdown("*Result contains symbolic expressions*")
            
    except Exception as e:
        st.error(f"Error computing scalar surface integral: {str(e)}")

def compute_flux_integral(F_x, F_y, F_z, surface_equation, x_min, x_max, y_min, y_max):
    """Compute flux integral through surface"""
    try:
        # Parse expressions
        fx_expr = expression_parser.parse(F_x) if F_x else 0
        fy_expr = expression_parser.parse(F_y) if F_y else 0
        fz_expr = expression_parser.parse(F_z) if F_z else 0
        z_expr = expression_parser.parse(surface_equation)
        
        x, y, z = sp.symbols('x y z')
        
        # Substitute surface equation into vector field
        fx_surface = fx_expr.subs(z, z_expr)
        fy_surface = fy_expr.subs(z, z_expr)
        fz_surface = fz_expr.subs(z, z_expr)
        
        # Compute normal vector: n̂ = (-∂z/∂x, -∂z/∂y, 1)
        dz_dx = sp.diff(z_expr, x)
        dz_dy = sp.diff(z_expr, y)
        
        # F⃗·n̂ = -Fx(∂z/∂x) - Fy(∂z/∂y) + Fz
        flux_density = -fx_surface * dz_dx - fy_surface * dz_dy + fz_surface
        
        # Compute double integral
        result = sp.integrate(flux_density, (x, x_min, x_max), (y, y_min, y_max))
        
        st.markdown(f"**Normal vector:** `n̂ = ({-dz_dx}, {-dz_dy}, 1)`")
        st.markdown(f"**F⃗·n̂:** `{flux_density}`")
        st.markdown(f"**Result:** `{result}`")
        
        # Try numerical evaluation
        try:
            numeric_result = float(result)
            st.markdown(f"**Numeric Value:** `{numeric_result:.6f}`")
        except:
            st.markdown("*Result contains symbolic expressions*")
            
    except Exception as e:
        st.error(f"Error computing flux integral: {str(e)}")

def verify_greens_theorem(P_expr, Q_expr):
    """Verify Green's theorem for a simple closed curve"""
    try:
        # Parse expressions
        P = expression_parser.parse(P_expr)
        Q = expression_parser.parse(Q_expr)
        
        x, y = sp.symbols('x y')
        
        # Compute curl (∂Q/∂x - ∂P/∂y)
        curl_2d = sp.diff(Q, x) - sp.diff(P, y)
        
        st.markdown(f"**P(x,y):** `{P}`")
        st.markdown(f"**Q(x,y):** `{Q}`")
        st.markdown(f"**∂Q/∂x - ∂P/∂y:** `{curl_2d}`")
        
        # For unit circle as example
        st.markdown("**Example verification for unit circle:**")
        
        # Double integral over unit disk
        # Convert to polar coordinates: x = r cos θ, y = r sin θ
        r, theta = sp.symbols('r theta', real=True, positive=True)
        curl_polar = curl_2d.subs([(x, r*sp.cos(theta)), (y, r*sp.sin(theta))])
        
        # Jacobian for polar coordinates is r
        double_integral = sp.integrate(curl_polar * r, (r, 0, 1), (theta, 0, 2*sp.pi))
        
        st.markdown(f"**Double integral (unit disk):** `{double_integral}`")
        
        # Line integral around unit circle
        # Parametric: x = cos(t), y = sin(t), t ∈ [0, 2π]
        t = sp.Symbol('t')
        x_param = sp.cos(t)
        y_param = sp.sin(t)
        
        P_param = P.subs([(x, x_param), (y, y_param)])
        Q_param = Q.subs([(x, x_param), (y, y_param)])
        
        dx_dt = sp.diff(x_param, t)
        dy_dt = sp.diff(y_param, t)
        
        line_integrand = P_param * dx_dt + Q_param * dy_dt
        line_integral = sp.integrate(line_integrand, (t, 0, 2*sp.pi))
        
        st.markdown(f"**Line integral (unit circle):** `{line_integral}`")
        
        # Check if they're equal
        if sp.simplify(double_integral - line_integral) == 0:
            st.success("**Green's theorem verified!** Both integrals are equal.")
        else:
            st.info("**Computed both integrals.** Check if they're numerically equal.")
            
    except Exception as e:
        st.error(f"Error verifying Green's theorem: {str(e)}")

def render_optimization_section():
    """Render optimization section for multivariable functions"""
    st.markdown("**Multivariable Optimization**")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Function f(x,y,z,...):",
            placeholder="x**2 + y**2 - 2*x*y + 3*x",
            key="optimization_expr"
        )
    
    with col2:
        variables = st.text_input(
            "Variables:",
            value="x,y",
            key="optimization_vars"
        )
    
    if expression and variables:
        var_list = [v.strip() for v in variables.split(',') if v.strip()]
        
        if st.button("Find Critical Points", key="find_critical_multivar"):
            try:
                expr = expression_parser.parse(expression)
                
                # Compute gradient components
                gradient_eqs = []
                for var in var_list:
                    partial = sp.diff(expr, sp.Symbol(var))
                    gradient_eqs.append(partial)
                
                st.markdown("**Gradient Equations (set to zero):**")
                for i, eq in enumerate(gradient_eqs):
                    st.markdown(f"∂f/∂{var_list[i]} = `{eq}` = 0")
                
                # Solve the system of gradient equations
                symbols = [sp.Symbol(var) for var in var_list]
                critical_points = sp.solve(gradient_eqs, symbols)
                
                st.markdown("**Critical Points:**")
                if critical_points:
                    if isinstance(critical_points, dict):
                        # Single critical point
                        point_str = ", ".join([f"{var} = {critical_points[sp.Symbol(var)]}" for var in var_list])
                        st.markdown(f"• ({point_str})")
                    elif isinstance(critical_points, list):
                        # Multiple critical points
                        for i, point in enumerate(critical_points):
                            if isinstance(point, dict):
                                point_str = ", ".join([f"{var} = {point[sp.Symbol(var)]}" for var in var_list])
                                st.markdown(f"• Point {i+1}: ({point_str})")
                            else:
                                st.markdown(f"• Point {i+1}: {point}")
                    else:
                        st.markdown(f"• {critical_points}")
                else:
                    st.markdown("No critical points found or system is too complex to solve analytically")
                
                # Hessian matrix for classification
                if len(var_list) == 2:  # For 2D functions, compute Hessian determinant
                    st.markdown("---")
                    st.markdown("**Hessian Matrix Analysis (2D):**")
                    
                    x, y = sp.symbols('x y')
                    fxx = sp.diff(expr, x, 2)
                    fyy = sp.diff(expr, y, 2)
                    fxy = sp.diff(expr, x, y)
                    
                    st.markdown(f"f_xx = `{fxx}`")
                    st.markdown(f"f_yy = `{fyy}`")
                    st.markdown(f"f_xy = `{fxy}`")
                    
                    # Hessian determinant
                    hessian_det = fxx * fyy - fxy**2
                    st.markdown(f"Hessian determinant = `{hessian_det}`")
                    
                    st.markdown("**Classification:** Use the second derivative test:")
                    st.markdown("• If D > 0 and f_xx > 0: Local minimum")
                    st.markdown("• If D > 0 and f_xx < 0: Local maximum")
                    st.markdown("• If D < 0: Saddle point")
                    st.markdown("• If D = 0: Test is inconclusive")
                
            except Exception as e:
                st.error(f"Error in optimization analysis: {str(e)}")
        
        # Constrained optimization (Lagrange multipliers)
        st.markdown("---")
        st.markdown("**Constrained Optimization (Lagrange Multipliers)**")
        
        constraint = st.text_input(
            "Constraint g(x,y,z,...) = 0:",
            placeholder="x**2 + y**2 - 1",
            key="constraint_expr"
        )
        
        if constraint and st.button("Apply Lagrange Multipliers", key="lagrange_multipliers"):
            try:
                f_expr = expression_parser.parse(expression)
                g_expr = expression_parser.parse(constraint)
                
                # Create Lagrangian: L = f - λg
                lam = sp.Symbol('lambda')
                symbols = [sp.Symbol(var) for var in var_list]
                all_symbols = symbols + [lam]
                
                # Lagrangian function
                lagrangian = f_expr - lam * g_expr
                
                st.markdown(f"**Lagrangian:** L = f - λg = `{lagrangian}`")
                
                # Compute gradient of Lagrangian
                lagrange_eqs = []
                
                # ∇f = λ∇g (gradient equations)
                for var in var_list:
                    var_sym = sp.Symbol(var)
                    eq = sp.diff(f_expr, var_sym) - lam * sp.diff(g_expr, var_sym)
                    lagrange_eqs.append(eq)
                
                # Add constraint equation g = 0
                lagrange_eqs.append(g_expr)
                
                st.markdown("**Lagrange Equations:**")
                for i, eq in enumerate(lagrange_eqs[:-1]):
                    st.markdown(f"∇f_{var_list[i]} = λ∇g_{var_list[i]}: `{eq}` = 0")
                st.markdown(f"Constraint: `{lagrange_eqs[-1]}` = 0")
                
                # Solve the Lagrange system
                try:
                    solutions = sp.solve(lagrange_eqs, all_symbols)
                    
                    st.markdown("**Critical Points on Constraint:**")
                    if solutions:
                        if isinstance(solutions, dict):
                            # Single solution
                            point_values = [solutions.get(sp.Symbol(var), 'N/A') for var in var_list]
                            lambda_val = solutions.get(lam, 'N/A')
                            point_str = ", ".join([f"{var} = {val}" for var, val in zip(var_list, point_values)])
                            st.markdown(f"• ({point_str}), λ = {lambda_val}")
                            
                            # Evaluate function at critical point
                            try:
                                subs_dict = {sp.Symbol(var): solutions[sp.Symbol(var)] for var in var_list if sp.Symbol(var) in solutions}
                                f_value = f_expr.subs(subs_dict)
                                st.markdown(f"  Function value: f = `{f_value}`")
                            except:
                                pass
                                
                        elif isinstance(solutions, list):
                            # Multiple solutions
                            for i, sol in enumerate(solutions):
                                if isinstance(sol, dict):
                                    point_values = [sol.get(sp.Symbol(var), 'N/A') for var in var_list]
                                    lambda_val = sol.get(lam, 'N/A')
                                    point_str = ", ".join([f"{var} = {val}" for var, val in zip(var_list, point_values)])
                                    st.markdown(f"• Point {i+1}: ({point_str}), λ = {lambda_val}")
                                    
                                    # Evaluate function at critical point
                                    try:
                                        subs_dict = {sp.Symbol(var): sol[sp.Symbol(var)] for var in var_list if sp.Symbol(var) in sol}
                                        f_value = f_expr.subs(subs_dict)
                                        st.markdown(f"  Function value: f = `{f_value}`")
                                    except:
                                        pass
                                else:
                                    st.markdown(f"• Point {i+1}: {sol}")
                        else:
                            st.markdown(f"• {solutions}")
                    else:
                        st.warning("No solutions found. The constraint may be incompatible or the system may be too complex.")
                        
                except Exception as solve_error:
                    st.warning(f"Could not solve the Lagrange system analytically: {str(solve_error)}")
                    st.info("Try simplifying the expressions or using numerical methods.")
                
            except Exception as e:
                st.error(f"Error in Lagrange multiplier analysis: {str(e)}")
        else:
            st.info("Please enter both a function and constraint to use Lagrange multipliers")

def render_differential_equations_section():
    """Render differential equations section"""
    st.subheader("Differential Equations Solver")
    st.markdown("Advanced ODE solver with multiple numerical methods")
    
    # Initialize solver
    ode_solver = AdvancedODESolver()
    
    # Equation type selection
    eq_type = st.selectbox(
        "Equation Type",
        ["First Order ODE", "Second Order ODE", "System of ODEs", "Boundary Value Problem", "Partial Differential Equation"]
    )
    
    if eq_type == "First Order ODE":
        render_first_order_ode(ode_solver)
    elif eq_type == "Second Order ODE":
        render_second_order_ode(ode_solver)
    elif eq_type == "System of ODEs":
        render_system_odes(ode_solver)
    elif eq_type == "Boundary Value Problem":
        render_bvp(ode_solver)
    else:
        render_pde(ode_solver)

def render_first_order_ode(solver):
    """Render first order ODE solver"""
    col1, col2 = st.columns(2)
    
    with col1:
        # ODE input
        st.markdown("**Differential Equation**")
        ode_str = st.text_input(
            "dy/dt = f(t, y)",
            value="-0.5 * y",
            help="Enter the right-hand side of dy/dt = f(t, y)"
        )
        
        # Initial conditions
        st.markdown("**Initial Conditions**")
        y0 = st.number_input("y(t₀)", value=1.0, step=0.1)
        t0 = st.number_input("t₀", value=0.0, step=0.1)
        
    with col2:
        # Solution parameters
        st.markdown("**Solution Parameters**")
        tf = st.number_input("Final time", value=10.0, min_value=t0+0.1, step=0.5)
        
        # Method selection
        method = st.selectbox(
            "Numerical Method",
            [m.value for m in SolverMethod]
        )
        
        num_points = st.slider("Number of points", 100, 5000, 1000)
    
    if st.button("Solve ODE", key="solve_first_order"):
        try:
            # Create ODE function
            def ode_func(t, y):
                # Create local variables for evaluation
                local_dict = {'t': t, 'y': y, 'np': np, 'exp': np.exp, 'sin': np.sin, 'cos': np.cos}
                return eval(ode_str, {"__builtins__": {}}, local_dict)
            
            # Create problem
            problem = ODEProblem(
                equation=ode_func,
                initial_conditions={'y0': y0},
                domain=(t0, tf),
                ode_type=ODEType.FIRST_ORDER
            )
            
            # Solve
            solution = solver.solve_ode(
                problem,
                method=SolverMethod[method],
                num_points=num_points
            )
            
            if solution.success:
                # Display solution
                st.success(f"Solution computed successfully using {method}")
                
                # Plot solution
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=solution.t,
                    y=solution.y,
                    mode='lines',
                    name='y(t)',
                    line=dict(width=2)
                ))
                
                fig.update_layout(
                    title="Solution of dy/dt = " + ode_str,
                    xaxis_title="t",
                    yaxis_title="y(t)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display some solution values
                with st.expander("Solution Values"):
                    indices = np.linspace(0, len(solution.t)-1, min(10, len(solution.t)), dtype=int)
                    data = {
                        "t": [solution.t[i] for i in indices],
                        "y(t)": [solution.y[i] for i in indices]
                    }
                    st.table(data)
                
                # Error estimate if available
                if solution.error_estimate is not None:
                    st.info(f"Estimated error: {np.max(np.abs(solution.error_estimate)):.2e}")
            else:
                st.error(f"Failed to solve ODE: {solution.message}")
                
        except Exception as e:
            st.error(f"Error solving ODE: {str(e)}")

def render_second_order_ode(solver):
    """Render second order ODE solver"""
    st.info("Convert second order ODE y'' = f(t, y, y') to system of first order ODEs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ODE input
        st.markdown("**Differential Equation**")
        ode_str = st.text_input(
            "y'' = f(t, y, y')",
            value="-y - 0.1*dy",
            help="Enter expression for y''. Use 'y' for y and 'dy' for y'"
        )
        
        # Initial conditions
        st.markdown("**Initial Conditions**")
        y0 = st.number_input("y(0)", value=1.0, step=0.1)
        dy0 = st.number_input("y'(0)", value=0.0, step=0.1)
        
    with col2:
        # Solution parameters
        st.markdown("**Solution Parameters**")
        tf = st.number_input("Final time", value=20.0, min_value=0.1, step=0.5)
        
        # Method selection
        method = st.selectbox(
            "Numerical Method",
            ["RK45", "DOP853", "BDF", "RADAU"],
            key="method_second"
        )
    
    if st.button("Solve Second Order ODE", key="solve_second_order"):
        try:
            # Convert to system of first order ODEs
            def system(t, state):
                y, dy = state
                # Evaluate second derivative
                local_dict = {'t': t, 'y': y, 'dy': dy, 'np': np, 'exp': np.exp, 'sin': np.sin, 'cos': np.cos}
                d2y = eval(ode_str, {"__builtins__": {}}, local_dict)
                return [dy, d2y]
            
            # Solve system
            solution = solver.solve_system(
                [lambda t, s: s[1], lambda t, s: system(t, s)[1]],
                initial_conditions=np.array([y0, dy0]),
                t_span=(0, tf),
                method=method
            )
            
            if solution.success:
                st.success(f"Solution computed successfully using {method}")
                
                # Plot solution
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=solution.t,
                    y=solution.y[:, 0],
                    mode='lines',
                    name='y(t)',
                    line=dict(width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=solution.t,
                    y=solution.y[:, 1],
                    mode='lines',
                    name="y'(t)",
                    line=dict(width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title="Solution of y'' = " + ode_str,
                    xaxis_title="t",
                    yaxis_title="Value",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Phase portrait
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=solution.y[:, 0],
                    y=solution.y[:, 1],
                    mode='lines',
                    name='Phase Portrait',
                    line=dict(width=2)
                ))
                
                fig2.update_layout(
                    title="Phase Portrait",
                    xaxis_title="y",
                    yaxis_title="y'",
                    hovermode='closest'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error(f"Failed to solve ODE: {solution.message}")
                
        except Exception as e:
            st.error(f"Error solving ODE: {str(e)}")

def render_system_odes(solver):
    """Render system of ODEs solver"""
    st.markdown("**System of ODEs**")
    
    # Predefined examples
    example = st.selectbox(
        "Select Example",
        ["Custom", "Lorenz System", "Van der Pol Oscillator", "Predator-Prey Model"]
    )
    
    if example == "Lorenz System":
        st.latex(r"\begin{cases} \dot{x} = \sigma(y - x) \\ \dot{y} = x(\rho - z) - y \\ \dot{z} = xy - \beta z \end{cases}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sigma = st.number_input("σ", value=10.0, step=0.5)
        with col2:
            rho = st.number_input("ρ", value=28.0, step=0.5)
        with col3:
            beta = st.number_input("β", value=8/3, step=0.1)
        
        initial = st.text_input("Initial conditions [x, y, z]", value="[1.0, 1.0, 1.0]")
        
        if st.button("Solve Lorenz System"):
            try:
                from core.advanced_ode_solver import lorenz_system
                
                # Parse initial conditions
                init_vals = eval(initial)
                
                # Create system function
                def system_func(t, state):
                    return lorenz_system(t, state, sigma, rho, beta)
                
                # Solve
                solution = solver.solve_system(
                    [lambda t, s: system_func(t, s)[i] for i in range(3)],
                    initial_conditions=np.array(init_vals),
                    t_span=(0, 50),
                    method='RK45'
                )
                
                if solution.success:
                    st.success("Lorenz system solved successfully!")
                    
                    # 3D trajectory
                    fig = go.Figure()
                    fig.add_trace(go.Scatter3d(
                        x=solution.y[:, 0],
                        y=solution.y[:, 1],
                        z=solution.y[:, 2],
                        mode='lines',
                        line=dict(width=2, color=solution.t, colorscale='Viridis'),
                        name='Trajectory'
                    ))
                    
                    fig.update_layout(
                        title="Lorenz Attractor",
                        scene=dict(
                            xaxis_title="x",
                            yaxis_title="y",
                            zaxis_title="z"
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Time series
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=solution.t, y=solution.y[:, 0], name='x(t)'))
                    fig2.add_trace(go.Scatter(x=solution.t, y=solution.y[:, 1], name='y(t)'))
                    fig2.add_trace(go.Scatter(x=solution.t, y=solution.y[:, 2], name='z(t)'))
                    
                    fig2.update_layout(
                        title="Time Series",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error solving system: {str(e)}")
    
    elif example == "Van der Pol Oscillator":
        st.latex(r"\begin{cases} \dot{x} = y \\ \dot{y} = \mu(1 - x^2)y - x \end{cases}")
        
        mu = st.number_input("μ", value=1.0, min_value=0.0, step=0.1)
        initial = st.text_input("Initial conditions [x, y]", value="[1.0, 0.0]")
        
        if st.button("Solve Van der Pol"):
            try:
                from core.advanced_ode_solver import van_der_pol
                
                # Parse initial conditions
                init_vals = eval(initial)
                
                # Create system function
                def system_func(t, state):
                    return van_der_pol(t, state, mu)
                
                # Solve
                solution = solver.solve_system(
                    [lambda t, s: system_func(t, s)[i] for i in range(2)],
                    initial_conditions=np.array(init_vals),
                    t_span=(0, 50),
                    method='RK45'
                )
                
                if solution.success:
                    st.success("Van der Pol oscillator solved successfully!")
                    
                    # Phase portrait
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=solution.y[:, 0],
                        y=solution.y[:, 1],
                        mode='lines',
                        line=dict(width=2),
                        name='Phase Portrait'
                    ))
                    
                    fig.update_layout(
                        title=f"Van der Pol Phase Portrait (μ = {mu})",
                        xaxis_title="x",
                        yaxis_title="y",
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Time series
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=solution.t, y=solution.y[:, 0], name='x(t)'))
                    
                    fig2.update_layout(
                        title="Position vs Time",
                        xaxis_title="Time",
                        yaxis_title="Position",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error solving system: {str(e)}")
    
    else:
        st.info("Custom system implementation - enter your system of ODEs")

def render_bvp(solver):
    """Render boundary value problem solver"""
    st.markdown("**Boundary Value Problem**")
    st.info("Solve second order BVP: y'' = f(x, y, y') with boundary conditions")
    
    # Not fully implemented in the basic UI
    st.warning("BVP solver interface coming soon. Use the API directly for now.")

def render_pde(solver):
    """Render partial differential equation solver"""
    st.markdown("**Partial Differential Equations**")
    
    pde_type = st.selectbox(
        "PDE Type",
        ["Heat Equation", "Wave Equation"]
    )
    
    if pde_type == "Heat Equation":
        st.latex(r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Parameters
            alpha = st.number_input("Thermal diffusivity α", value=1.0, min_value=0.1, step=0.1)
            L = st.number_input("Domain length", value=1.0, min_value=0.1, step=0.1)
            
            # Initial condition
            ic_type = st.selectbox("Initial condition", ["Sine wave", "Step function", "Gaussian"])
            
        with col2:
            # Boundary conditions
            bc_left = st.number_input("Left boundary (u(0,t))", value=0.0, step=0.1)
            bc_right = st.number_input("Right boundary (u(L,t))", value=0.0, step=0.1)
            
            # Time
            t_max = st.number_input("Final time", value=0.5, min_value=0.01, step=0.05)
        
        if st.button("Solve Heat Equation"):
            try:
                # Define initial condition
                if ic_type == "Sine wave":
                    ic_func = lambda x: np.sin(np.pi * x / L)
                elif ic_type == "Step function":
                    ic_func = lambda x: np.where(x < L/2, 1.0, 0.0)
                else:  # Gaussian
                    ic_func = lambda x: np.exp(-50*(x - L/2)**2)
                
                # Solve PDE
                solution = solver.solve_pde_heat_equation(
                    initial_condition=ic_func,
                    boundary_conditions=(bc_left, bc_right),
                    x_domain=(0, L),
                    t_domain=(0, t_max),
                    nx=100,
                    nt=1000,
                    alpha=alpha
                )
                
                st.success("Heat equation solved successfully!")
                
                # Create animation frames
                frames = []
                for i in range(0, len(solution['t']), max(1, len(solution['t'])//50)):
                    frames.append(go.Frame(
                        data=[go.Scatter(
                            x=solution['x'],
                            y=solution['u'][i, :],
                            mode='lines',
                            line=dict(width=2)
                        )],
                        name=f"t={solution['t'][i]:.3f}"
                    ))
                
                # Create figure
                fig = go.Figure(
                    data=[go.Scatter(
                        x=solution['x'],
                        y=solution['u'][0, :],
                        mode='lines',
                        line=dict(width=2)
                    )],
                    frames=frames
                )
                
                # Add play button
                fig.update_layout(
                    title="Heat Equation Solution",
                    xaxis_title="x",
                    yaxis_title="u(x,t)",
                    updatemenus=[{
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {'label': 'Play', 'method': 'animate', 'args': [None, {'frame': {'duration': 50}}]},
                            {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                        ]
                    }],
                    sliders=[{
                        'active': 0,
                        'steps': [{'args': [[f.name], {'frame': {'duration': 0}, 'mode': 'immediate'}], 
                                  'label': f"t={solution['t'][i]:.3f}", 'method': 'animate'} 
                                 for i, f in enumerate(frames)]
                    }]
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                if solution['stability_parameter'] > 0.5:
                    st.warning(f"Warning: Stability parameter r = {solution['stability_parameter']:.3f} > 0.5. Solution may be unstable.")
                
            except Exception as e:
                st.error(f"Error solving PDE: {str(e)}")
