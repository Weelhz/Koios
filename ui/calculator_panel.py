import streamlit as st
import sympy as sp
import numpy as np
from core.calculation_engine import calculation_engine
from core.expression_parser import expression_parser
import math

def render_calculator_panel():
    """Render the scientific calculator panel"""
    st.header("Scientific Calculator")
    st.markdown("*Advanced mathematical expression evaluator with step-by-step solutions*")
    
    # Calculator mode selection
    # Initialize session state for expression
    if 'calc_expression' not in st.session_state:
        st.session_state.calc_expression = ""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        expression = st.text_input(
            "Enter mathematical expression:",
            value=st.session_state.calc_expression,
            placeholder="e.g., sin(pi/4) + log(e^2) + sqrt(16)",
            help="Use standard mathematical notation. Supported functions: sin, cos, tan, log, ln, exp, sqrt, etc.",
            key="main_expression"
        )
        # Update session state when user types
        st.session_state.calc_expression = expression
    
    with col2:
        mode = st.selectbox(
            "Mode:",
            ["Evaluate", "Simplify", "Expand", "Factor"]
        )
    
    # Quick function buttons
    st.markdown("**Quick Functions:**")
    button_cols = st.columns(8)
    
    quick_functions = [
        ("π", "pi"), ("e", "e"), ("√", "sqrt("), ("ln", "ln("),
        ("sin", "sin("), ("cos", "cos("), ("tan", "tan("), ("∫", "integrate(")
    ]
    
    for i, (display, func) in enumerate(quick_functions):
        with button_cols[i]:
            if st.button(display, key=f"btn_{i}"):
                st.session_state.calc_expression += func
                st.rerun()
    
    # Variable substitution section
    st.markdown("---")
    st.subheader("Variable Substitution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        var_names = st.text_input(
            "Variables (comma-separated):",
            placeholder="x, y, z",
            help="Enter variable names separated by commas"
        )
    
    with col2:
        var_values = st.text_input(
            "Values (comma-separated):",
            placeholder="1, 2, 3",
            help="Enter corresponding values separated by commas"
        )
    
    # Process calculation
    if expression:
        # Parse variables if provided
        variables = None
        if var_names and var_values:
            try:
                names = [name.strip() for name in var_names.split(',')]
                values = [float(val.strip()) for val in var_values.split(',')]
                if len(names) == len(values):
                    variables = dict(zip(names, values))
                else:
                    st.warning("Number of variable names and values must match!")
            except ValueError:
                st.warning("Invalid variable values. Please enter numeric values.")
        
        # Perform calculation based on mode
        if mode == "Evaluate":
            result = calculation_engine.evaluate_expression(expression, variables)
            display_calculation_result(result, "Evaluation")
        
        elif mode == "Simplify":
            result = calculation_engine.simplify_expression(expression)
            display_simplification_result(result)
        
        elif mode == "Expand":
            result = calculation_engine.expand_expression(expression)
            display_expansion_result(result)
        
        elif mode == "Factor":
            result = calculation_engine.factor_expression(expression)
            display_factoring_result(result)
        
        # Expression validation and info
        st.markdown("---")
        st.subheader("Expression Analysis")
        validation = expression_parser.validate_expression(expression)
        display_expression_info(validation)

def display_calculation_result(result, operation_name):
    """Display calculation results"""
    if result['success']:
        st.success(f"✅ {operation_name} completed successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Symbolic Result:**")
            if result['symbolic_result'] is not None:
                st.latex(sp.latex(result['symbolic_result']))
            else:
                st.write("No symbolic result")
        
        with col2:
            st.markdown("**Numeric Result:**")
            if result['numeric_result'] is not None:
                st.metric("Value", f"{result['numeric_result']:.10g}")
                
                # Additional numeric info
                if isinstance(result['numeric_result'], (int, float)):
                    st.write(f"Scientific notation: {result['numeric_result']:.6e}")
                    if result['numeric_result'] != 0:
                        st.write(f"Magnitude: {abs(result['numeric_result']):.6e}")
            else:
                st.write("No numeric result available")
    
    else:
        st.error(f"❌ {operation_name} failed: {result['error']}")

def display_simplification_result(result):
    """Display simplification results"""
    if result['success']:
        st.success("✅ Expression simplified successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original:**")
            st.latex(sp.latex(result['original']))
        
        with col2:
            st.markdown("**Simplified:**")
            st.latex(sp.latex(result['simplified']))
        
        # Check if simplification made a difference
        if str(result['original']) == str(result['simplified']):
            st.info("ℹ️ Expression is already in its simplest form")
    
    else:
        st.error(f"❌ Simplification failed: {result['error']}")

def display_expansion_result(result):
    """Display expansion results"""
    if result['success']:
        st.success("✅ Expression expanded successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original:**")
            st.latex(sp.latex(result['original']))
        
        with col2:
            st.markdown("**Expanded:**")
            st.latex(sp.latex(result['expanded']))
        
        # Check if expansion made a difference
        if str(result['original']) == str(result['expanded']):
            st.info("ℹ️ Expression is already expanded")
    
    else:
        st.error(f"❌ Expansion failed: {result['error']}")

def display_factoring_result(result):
    """Display factoring results"""
    if result['success']:
        st.success("✅ Expression factored successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original:**")
            st.latex(sp.latex(result['original']))
        
        with col2:
            st.markdown("**Factored:**")
            st.latex(sp.latex(result['factored']))
        
        # Check if factoring made a difference
        if str(result['original']) == str(result['factored']):
            st.info("ℹ️ Expression cannot be factored further")
    
    else:
        st.error(f"❌ Factoring failed: {result['error']}")

def display_expression_info(validation):
    """Display expression validation and information"""
    if validation['valid']:
        st.success("✅ Expression is valid")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Variables:**")
            if validation['variables']:
                for var in validation['variables']:
                    st.write(f"• {var}")
            else:
                st.write("No variables")
        
        with col2:
            st.markdown("**Functions:**")
            if validation['functions']:
                for func in validation['functions']:
                    st.write(f"• {func}")
            else:
                st.write("No functions")
        
        with col3:
            if validation['expression']:
                info = expression_parser.get_expression_info(validation['expression'])
                st.markdown("**Properties:**")
                st.write(f"• Complexity: {info['complexity']}")
                st.write(f"• Polynomial: {info['is_polynomial']}")
                st.write(f"• Rational: {info['is_rational']}")
    
    else:
        st.error(f"❌ Invalid expression: {validation['error']}")
        st.markdown("**Help:**")
        st.markdown("""
        - Use standard mathematical notation
        - Supported functions: sin, cos, tan, asin, acos, atan, ln, log, exp, sqrt, abs
        - Constants: pi, e
        - Use ** for exponentiation (e.g., x**2)
        - Use parentheses for grouping
        """)
