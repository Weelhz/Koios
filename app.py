import streamlit as st
import sys
import os
from pathlib import Path

# Add all module paths
current_dir = Path(__file__).parent
sys.path.extend([
    str(current_dir / "core"),
    str(current_dir / "ui"),
    str(current_dir / "utils")
])

# Import core modules
try:
    from ui.calculator_panel import render_calculator_panel
    from ui.matrix_panel import render_matrix_panel
    from ui.calculus_panel import render_calculus_panel
    from ui.equation_solver_panel import render_equation_solver_panel
    from ui.physics_panel import render_physics_panel
    from ui.visualization_panel import render_visualization_panel
    from ui.engineering_panel import render_engineering_panel
    from ui.complex_analysis_panel import render_complex_analysis_panel
    from ui.tensor_calculus_panel import render_tensor_calculus_panel
    from ui.numerical_methods_panel import render_numerical_methods_panel
    from ui.optimization_panel import render_optimization_panel
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Error loading modules: {e}")
    MODULES_LOADED = False

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Koios - Advanced Mathematical Toolset",
        page_icon="K",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("Koios - Advanced Mathematical Toolset")
    st.markdown("*Comprehensive mathematical computation platform with advanced capabilities*")
    
    if not MODULES_LOADED:
        st.error("Failed to load core modules. Please check the installation.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Tool categories
    tools = {
        "Basic Tools": {
            "Scientific Calculator": "calculator",
            "Matrix Calculator": "matrix"
        },
        "Advanced Tools": {
            "Calculus Tools": "calculus",
            "Equation Solver": "equations",
            "Complex Analysis": "complex",
            "Tensor Calculus": "tensor",
            "Numerical Methods": "numerical",
            "Optimization": "optimization"
        },
        "Physics": {
            "Physics Simulations": "physics"
        },
        "Visualization": {
            "Function Plotting": "visualization"
        },
        "Engineering": {
            "Advanced Simulations": "engineering"
        }
    }
    
    # Create tool selection
    selected_category = st.sidebar.selectbox(
        "Select Category:",
        options=list(tools.keys())
    )
    
    selected_tool_name = st.sidebar.selectbox(
        "Select Tool:",
        options=list(tools[selected_category].keys())
    )
    
    tool_key = tools[selected_category][selected_tool_name]
    
    st.sidebar.markdown("---")
    
    # Main content area
    try:
        if tool_key == "calculator":
            render_calculator_panel()
            
        elif tool_key == "matrix":
            render_matrix_panel()
            
        elif tool_key == "calculus":
            render_calculus_panel()
            
        elif tool_key == "equations":
            render_equation_solver_panel()
            
        elif tool_key == "complex":
            render_complex_analysis_panel()
            
        elif tool_key == "physics":
            render_physics_panel()
            
        elif tool_key == "visualization":
            render_visualization_panel()
            
        elif tool_key == "engineering":
            render_engineering_panel()
            
        elif tool_key == "tensor":
            render_tensor_calculus_panel()
            
        elif tool_key == "numerical":
            render_numerical_methods_panel()
            
        elif tool_key == "optimization":
            render_optimization_panel()
            
    except Exception as e:
        st.error(f"Error rendering panel: {str(e)}")
        st.exception(e)
    


if __name__ == "__main__":
    main()