import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import scipy.optimize as sp_opt
from scipy.stats import norm
import warnings

class OptimizationType(Enum):
    """Types of optimization problems"""
    UNCONSTRAINED = "UNCONSTRAINED"
    CONSTRAINED = "CONSTRAINED"
    MULTI_OBJECTIVE = "MULTI_OBJECTIVE"
    ROBUST = "ROBUST"
    STOCHASTIC = "STOCHASTIC"

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    x_opt: np.ndarray
    f_opt: float
    iterations: int
    converged: bool
    constraint_violation: Optional[float] = None
    pareto_front: Optional[np.ndarray] = None
    uncertainty_bounds: Optional[Tuple[float, float]] = None
    history: Optional[Dict[str, List]] = None

class OptimizationAlgorithmsEngine:
    """Advanced optimization algorithms engine"""
    
    def __init__(self):
        self.tolerance = 1e-8
        self.max_iterations = 1000
        self.population_size = 50
        
    def lbfgs_b(self, f: Callable, grad_f: Callable, x0: np.ndarray,
                bounds: Optional[List[Tuple[float, float]]] = None,
                m: int = 10) -> OptimizationResult:
        """
        Limited-memory BFGS with bound constraints
        
        Args:
            f: Objective function
            grad_f: Gradient function
            x0: Initial point
            bounds: Variable bounds [(low, high), ...]
            m: Number of corrections to approximate Hessian
        """
        n = len(x0)
        x = x0.copy()
        
        # Initialize L-BFGS memory
        s_list = []  # x differences
        y_list = []  # gradient differences
        
        # History tracking
        history = {'f_vals': [f(x)], 'grad_norms': []}
        
        g = grad_f(x)
        history['grad_norms'].append(np.linalg.norm(g))
        
        for k in range(self.max_iterations):
            if np.linalg.norm(g) < self.tolerance:
                break
            
            # Compute search direction using L-BFGS two-loop recursion
            q = g.copy()
            alpha_list = []
            
            # First loop
            for i in range(len(s_list) - 1, -1, -1):
                rho_i = 1.0 / np.dot(y_list[i], s_list[i])
                alpha_i = rho_i * np.dot(s_list[i], q)
                q = q - alpha_i * y_list[i]
                alpha_list.append(alpha_i)
            
            # Scaling
            if len(s_list) > 0:
                gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
                r = gamma * q
            else:
                r = q
            
            # Second loop
            alpha_list.reverse()
            for i in range(len(s_list)):
                rho_i = 1.0 / np.dot(y_list[i], s_list[i])
                beta = rho_i * np.dot(y_list[i], r)
                r = r + s_list[i] * (alpha_list[i] - beta)
            
            # Search direction
            d = -r
            
            # Line search with bound constraints
            alpha = self._bounded_line_search(f, grad_f, x, d, bounds)
            
            # Update
            x_new = x + alpha * d
            
            # Apply bounds
            if bounds:
                for i, (low, high) in enumerate(bounds):
                    x_new[i] = np.clip(x_new[i], low, high)
            
            g_new = grad_f(x_new)
            
            # Update L-BFGS memory
            s = x_new - x
            y = g_new - g
            
            if np.dot(s, y) > 1e-10:  # Curvature condition
                s_list.append(s)
                y_list.append(y)
                
                # Limit memory
                if len(s_list) > m:
                    s_list.pop(0)
                    y_list.pop(0)
            
            # Move to next iteration
            x = x_new
            g = g_new
            
            history['f_vals'].append(f(x))
            history['grad_norms'].append(np.linalg.norm(g))
        
        return OptimizationResult(
            x_opt=x,
            f_opt=f(x),
            iterations=k + 1,
            converged=np.linalg.norm(g) < self.tolerance,
            history=history
        )
    
    def _bounded_line_search(self, f: Callable, grad_f: Callable, 
                           x: np.ndarray, d: np.ndarray,
                           bounds: Optional[List[Tuple[float, float]]]) -> float:
        """Line search with bound constraints"""
        alpha_max = 1.0
        
        # Find maximum step size that satisfies bounds
        if bounds:
            for i, (low, high) in enumerate(bounds):
                if d[i] > 0:
                    alpha_max = min(alpha_max, (high - x[i]) / d[i])
                elif d[i] < 0:
                    alpha_max = min(alpha_max, (low - x[i]) / d[i])
        
        # Armijo line search
        alpha = min(1.0, alpha_max)
        c1 = 1e-4
        rho = 0.8
        
        f0 = f(x)
        g0 = np.dot(grad_f(x), d)
        
        while alpha > 1e-10:
            if f(x + alpha * d) <= f0 + c1 * alpha * g0:
                break
            alpha *= rho
        
        return alpha
    
    def genetic_algorithm(self, f: Callable, bounds: List[Tuple[float, float]],
                         pop_size: Optional[int] = None,
                         crossover_prob: float = 0.8,
                         mutation_prob: float = 0.1) -> OptimizationResult:
        """
        Genetic algorithm for global optimization
        
        Args:
            f: Objective function
            bounds: Variable bounds
            pop_size: Population size
            crossover_prob: Crossover probability
            mutation_prob: Mutation probability
        """
        if pop_size is None:
            pop_size = self.population_size
        
        n_vars = len(bounds)
        
        # Initialize population
        population = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(pop_size, n_vars)
        )
        
        # Evaluate initial population
        fitness = np.array([f(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        history = {'best_fitness': [best_fitness], 'avg_fitness': [np.mean(fitness)]}
        
        for generation in range(self.max_iterations):
            # Selection (tournament)
            new_population = []
            
            for _ in range(pop_size):
                # Tournament selection
                idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
                if fitness[idx1] < fitness[idx2]:
                    parent1 = population[idx1]
                else:
                    parent1 = population[idx2]
                
                idx3, idx4 = np.random.choice(pop_size, 2, replace=False)
                if fitness[idx3] < fitness[idx4]:
                    parent2 = population[idx3]
                else:
                    parent2 = population[idx4]
                
                if np.random.random() < crossover_prob:
                    mask = np.random.random(n_vars) < 0.5
                    child = np.where(mask, parent1, parent2)
                else:
                    child = parent1.copy()
                
                for i in range(n_vars):
                    if np.random.random() < mutation_prob:
                        child[i] += np.random.normal(0, 0.1 * (bounds[i][1] - bounds[i][0]))
                        child[i] = np.clip(child[i], bounds[i][0], bounds[i][1])
                
                new_population.append(child)
            
            population = np.array(new_population)
            fitness = np.array([f(ind) for ind in population])
            
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_individual = population[min_idx].copy()
                best_fitness = fitness[min_idx]
            
            history['best_fitness'].append(best_fitness)
            history['avg_fitness'].append(np.mean(fitness))
            
            # Convergence check
            if generation > 100:
                recent_improvement = abs(history['best_fitness'][-1] - 
                                      history['best_fitness'][-100])
                if recent_improvement < self.tolerance:
                    break
        
        return OptimizationResult(
            x_opt=best_individual,
            f_opt=best_fitness,
            iterations=generation + 1,
            converged=True,
            history=history
        )
    
    def particle_swarm_optimization(self, f: Callable, bounds: List[Tuple[float, float]],
                                  swarm_size: Optional[int] = None,
                                  w: float = 0.7, c1: float = 1.5, c2: float = 1.5) -> OptimizationResult:
        """
        Particle Swarm Optimization (PSO)
        
        Args:
            f: Objective function
            bounds: Variable bounds
            swarm_size: Number of particles
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        if swarm_size is None:
            swarm_size = self.population_size
        
        n_vars = len(bounds)
        
        # Initialize swarm
        positions = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(swarm_size, n_vars)
        )
        
        velocities = np.random.uniform(
            low=[-0.1 * (b[1] - b[0]) for b in bounds],
            high=[0.1 * (b[1] - b[0]) for b in bounds],
            size=(swarm_size, n_vars)
        )
        
        # Initialize personal and global bests
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([f(p) for p in positions])
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]
        
        history = {'global_best': [global_best_score]}
        
        for iteration in range(self.max_iterations):
            # Update velocities and positions
            for i in range(swarm_size):
                # Random coefficients
                r1 = np.random.random(n_vars)
                r2 = np.random.random(n_vars)
                
                # Update velocity
                velocities[i] = (w * velocities[i] +
                               c1 * r1 * (personal_best_positions[i] - positions[i]) +
                               c2 * r2 * (global_best_position - positions[i]))
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Apply bounds
                for j in range(n_vars):
                    positions[i, j] = np.clip(positions[i, j], bounds[j][0], bounds[j][1])
                
                # Update personal best
                score = f(positions[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                    
                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()
            
            history['global_best'].append(global_best_score)
            
            # Convergence check
            if iteration > 50:
                recent_improvement = abs(history['global_best'][-1] - 
                                      history['global_best'][-50])
                if recent_improvement < self.tolerance:
                    break
        
        return OptimizationResult(
            x_opt=global_best_position,
            f_opt=global_best_score,
            iterations=iteration + 1,
            converged=True,
            history=history
        )
    
    def sequential_quadratic_programming(self, f: Callable, grad_f: Callable,
                                       constraints: List[Dict[str, Any]],
                                       x0: np.ndarray) -> OptimizationResult:
        """
        Sequential Quadratic Programming (SQP) for constrained optimization
        
        Args:
            f: Objective function
            grad_f: Gradient of objective
            constraints: List of constraint dicts with 'type', 'fun', 'jac'
            x0: Initial point
        """
        x = x0.copy()
        n = len(x)
        
        # Separate equality and inequality constraints
        eq_constraints = [c for c in constraints if c['type'] == 'eq']
        ineq_constraints = [c for c in constraints if c['type'] == 'ineq']
        
        m_eq = len(eq_constraints)
        m_ineq = len(ineq_constraints)
        
        # Lagrange multipliers
        lambda_eq = np.zeros(m_eq)
        lambda_ineq = np.zeros(m_ineq)
        
        history = {'f_vals': [f(x)], 'constraint_violations': []}
        
        for iteration in range(self.max_iterations):
            # Evaluate constraints and gradients
            g_eq = np.array([c['fun'](x) for c in eq_constraints]) if m_eq > 0 else np.array([])
            g_ineq = np.array([c['fun'](x) for c in ineq_constraints]) if m_ineq > 0 else np.array([])
            
            # Constraint violation
            violation = 0
            if m_eq > 0:
                violation += np.sum(np.abs(g_eq))
            if m_ineq > 0:
                violation += np.sum(np.maximum(0, g_ineq))
            
            history['constraint_violations'].append(violation)
            
            if violation < self.tolerance and iteration > 0:
                # Check KKT conditions
                grad_L = grad_f(x)
                if m_eq > 0:
                    for i, c in enumerate(eq_constraints):
                        grad_L += lambda_eq[i] * c['jac'](x)
                if m_ineq > 0:
                    for i, c in enumerate(ineq_constraints):
                        grad_L += lambda_ineq[i] * c['jac'](x)
                
                if np.linalg.norm(grad_L) < self.tolerance:
                    break
            
            # Set up QP subproblem
            # Approximate Hessian (BFGS update would go here)
            B = np.eye(n)  # Simplified - using identity
            
            # Gradients
            grad_f_x = grad_f(x)
            
            # Linearized constraints
            A_eq = np.array([c['jac'](x) for c in eq_constraints]) if m_eq > 0 else None
            b_eq = -g_eq if m_eq > 0 else None
            
            A_ineq = np.array([c['jac'](x) for c in ineq_constraints]) if m_ineq > 0 else None
            b_ineq = -g_ineq if m_ineq > 0 else None
            
            # Solve QP subproblem (simplified - would use QP solver)
            # min 0.5 * p.T @ B @ p + grad_f.T @ p
            # s.t. A_eq @ p = b_eq, A_ineq @ p <= b_ineq
            
            # Simple gradient step (full SQP would solve QP)
            p = -grad_f_x / np.linalg.norm(grad_f_x)
            
            # Line search
            alpha = 1.0
            merit_param = 10.0  # Penalty parameter
            
            def merit_function(x):
                result = f(x)
                if m_eq > 0:
                    result += merit_param * np.sum(np.abs([c['fun'](x) for c in eq_constraints]))
                if m_ineq > 0:
                    result += merit_param * np.sum(np.maximum(0, [c['fun'](x) for c in ineq_constraints]))
                return result
            
            merit_0 = merit_function(x)
            
            while alpha > 1e-10:
                x_new = x + alpha * p
                if merit_function(x_new) < merit_0:
                    break
                alpha *= 0.5
            
            x = x_new
            history['f_vals'].append(f(x))
        
        return OptimizationResult(
            x_opt=x,
            f_opt=f(x),
            iterations=iteration + 1,
            converged=violation < self.tolerance,
            constraint_violation=violation,
            history=history
        )
    
    def nsga_ii(self, objectives: List[Callable], bounds: List[Tuple[float, float]],
                pop_size: Optional[int] = None) -> OptimizationResult:
        """
        NSGA-II for multi-objective optimization
        
        Args:
            objectives: List of objective functions
            bounds: Variable bounds
            pop_size: Population size
        """
        if pop_size is None:
            pop_size = self.population_size
        
        n_vars = len(bounds)
        n_obj = len(objectives)
        
        # Initialize population
        population = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(pop_size, n_vars)
        )
        
        def evaluate_objectives(x):
            return np.array([obj(x) for obj in objectives])
        
        def non_dominated_sorting(pop, obj_vals):
            """Fast non-dominated sorting"""
            n = len(pop)
            domination_count = np.zeros(n)
            dominated_solutions = [[] for _ in range(n)]
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    
                    # Check if i dominates j
                    if all(obj_vals[i] <= obj_vals[j]) and any(obj_vals[i] < obj_vals[j]):
                        dominated_solutions[i].append(j)
                    elif all(obj_vals[j] <= obj_vals[i]) and any(obj_vals[j] < obj_vals[i]):
                        domination_count[i] += 1
                
                if domination_count[i] == 0:
                    fronts[0].append(i)
            
            i = 0
            while len(fronts[i]) > 0:
                next_front = []
                for sol in fronts[i]:
                    for dominated in dominated_solutions[sol]:
                        domination_count[dominated] -= 1
                        if domination_count[dominated] == 0:
                            next_front.append(dominated)
                i += 1
                fronts.append(next_front)
            
            return fronts[:-1]  # Remove empty last front
        
        def crowding_distance(obj_vals):
            """Calculate crowding distance"""
            n = len(obj_vals)
            distance = np.zeros(n)
            
            for m in range(n_obj):
                # Sort by objective m
                sorted_idx = np.argsort(obj_vals[:, m])
                
                # Boundary points have infinite distance
                distance[sorted_idx[0]] = np.inf
                distance[sorted_idx[-1]] = np.inf
                
                # Calculate distance for others
                obj_range = obj_vals[sorted_idx[-1], m] - obj_vals[sorted_idx[0], m]
                if obj_range > 0:
                    for i in range(1, n-1):
                        distance[sorted_idx[i]] += (
                            obj_vals[sorted_idx[i+1], m] - 
                            obj_vals[sorted_idx[i-1], m]
                        ) / obj_range
            
            return distance
        
        # Main NSGA-II loop
        pareto_front_history = []
        
        for generation in range(self.max_iterations // 10):  # Fewer iterations for multi-obj
            # Evaluate objectives
            obj_vals = np.array([evaluate_objectives(ind) for ind in population])
            
            # Create offspring
            offspring = []
            for _ in range(pop_size):
                # Tournament selection
                idx1, idx2 = np.random.choice(pop_size, 2, replace=False)
                parent1 = population[idx1]
                parent2 = population[idx2]
                
                # Crossover (SBX)
                child = np.zeros(n_vars)
                for i in range(n_vars):
                    if np.random.random() < 0.5:
                        # SBX crossover
                        beta = np.random.random()
                        child[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                    else:
                        child[i] = parent1[i] if np.random.random() < 0.5 else parent2[i]
                
                # Mutation
                for i in range(n_vars):
                    if np.random.random() < 0.1:
                        child[i] += np.random.normal(0, 0.1 * (bounds[i][1] - bounds[i][0]))
                    child[i] = np.clip(child[i], bounds[i][0], bounds[i][1])
                
                offspring.append(child)
            
            offspring = np.array(offspring)
            
            # Combine population and offspring
            combined_pop = np.vstack([population, offspring])
            combined_obj = np.array([evaluate_objectives(ind) for ind in combined_pop])
            
            # Non-dominated sorting
            fronts = non_dominated_sorting(combined_pop, combined_obj)
            
            # Select next population
            new_pop = []
            new_pop_idx = []
            
            for front in fronts:
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend(combined_pop[front])
                    new_pop_idx.extend(front)
                else:
                    # Use crowding distance for selection
                    remaining = pop_size - len(new_pop)
                    front_obj = combined_obj[front]
                    distances = crowding_distance(front_obj)
                    
                    # Sort by crowding distance
                    sorted_idx = np.argsort(distances)[::-1]
                    selected = [front[i] for i in sorted_idx[:remaining]]
                    
                    new_pop.extend(combined_pop[selected])
                    new_pop_idx.extend(selected)
                    break
            
            population = np.array(new_pop)
            
            # Store Pareto front
            pareto_front = combined_obj[fronts[0]]
            pareto_front_history.append(pareto_front)
        
        # Final Pareto front
        final_obj = np.array([evaluate_objectives(ind) for ind in population])
        fronts = non_dominated_sorting(population, final_obj)
        pareto_front = population[fronts[0]]
        pareto_obj = final_obj[fronts[0]]
        
        return OptimizationResult(
            x_opt=pareto_front[0],  # Return one solution
            f_opt=np.mean(pareto_obj[0]),  # Mean of objectives
            iterations=generation + 1,
            converged=True,
            pareto_front=pareto_front,
            history={'pareto_fronts': pareto_front_history}
        )
    
    def simulated_annealing(self, f: Callable, bounds: List[Tuple[float, float]],
                          x0: Optional[np.ndarray] = None,
                          T0: float = 100.0, cooling_rate: float = 0.95) -> OptimizationResult:
        """
        Simulated Annealing for global optimization
        
        Args:
            f: Objective function
            bounds: Variable bounds
            x0: Initial point (optional)
            T0: Initial temperature
            cooling_rate: Temperature cooling rate
        """
        n_vars = len(bounds)
        
        # Initialize
        if x0 is None:
            x = np.random.uniform(
                low=[b[0] for b in bounds],
                high=[b[1] for b in bounds]
            )
        else:
            x = x0.copy()
        
        best_x = x.copy()
        best_f = f(x)
        current_f = best_f
        
        T = T0
        history = {'best_f': [best_f], 'current_f': [current_f], 'temperature': [T]}
        
        for iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor = x.copy()
            
            # Random walk with temperature-dependent step size
            for i in range(n_vars):
                step_size = 0.1 * (bounds[i][1] - bounds[i][0]) * T / T0
                neighbor[i] += np.random.uniform(-step_size, step_size)
                neighbor[i] = np.clip(neighbor[i], bounds[i][0], bounds[i][1])
            
            # Evaluate neighbor
            neighbor_f = f(neighbor)
            
            # Accept or reject
            delta = neighbor_f - current_f
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                x = neighbor
                current_f = neighbor_f
                
                if current_f < best_f:
                    best_x = x.copy()
                    best_f = current_f
            
            # Cool down
            T *= cooling_rate
            
            # Record history
            history['best_f'].append(best_f)
            history['current_f'].append(current_f)
            history['temperature'].append(T)
            
            # Stop if frozen
            if T < 1e-10:
                break
        
        return OptimizationResult(
            x_opt=best_x,
            f_opt=best_f,
            iterations=iteration + 1,
            converged=True,
            history=history
        )
    
    def robust_optimization(self, f: Callable, bounds: List[Tuple[float, float]],
                          uncertainty_set: Dict[str, Any],
                          method: str = 'worst_case') -> OptimizationResult:
        """
        Robust optimization under uncertainty
        
        Args:
            f: Objective function f(x, xi) where xi is uncertain parameter
            bounds: Variable bounds
            uncertainty_set: Description of uncertainty
            method: 'worst_case' or 'expected_value'
        """
        n_vars = len(bounds)
        
        if method == 'worst_case':
            # Min-max robust optimization
            def robust_objective(x):
                # Sample uncertainty set
                n_samples = 100
                worst_val = -np.inf
                
                for _ in range(n_samples):
                    # Generate uncertain parameter
                    if uncertainty_set['type'] == 'box':
                        xi = np.random.uniform(
                            uncertainty_set['lower'],
                            uncertainty_set['upper']
                        )
                    elif uncertainty_set['type'] == 'ellipsoid':
                        # Sample from ellipsoid
                        xi = np.random.multivariate_normal(
                            uncertainty_set['center'],
                            uncertainty_set['shape']
                        )
                    
                    val = f(x, xi)
                    worst_val = max(worst_val, val)
                
                return worst_val
            
        else:  # expected_value
            # Stochastic optimization
            def robust_objective(x):
                n_samples = 100
                values = []
                
                for _ in range(n_samples):
                    if uncertainty_set['type'] == 'normal':
                        xi = np.random.normal(
                            uncertainty_set['mean'],
                            uncertainty_set['std']
                        )
                    else:
                        xi = np.random.uniform(
                            uncertainty_set['lower'],
                            uncertainty_set['upper']
                        )
                    
                    values.append(f(x, xi))
                
                return np.mean(values)
        
        # Use genetic algorithm for robust objective
        result = self.genetic_algorithm(robust_objective, bounds)
        
        # Compute uncertainty bounds
        x_opt = result.x_opt
        values = []
        for _ in range(1000):
            if uncertainty_set['type'] == 'normal':
                xi = np.random.normal(uncertainty_set['mean'], uncertainty_set['std'])
            else:
                xi = np.random.uniform(uncertainty_set['lower'], uncertainty_set['upper'])
            values.append(f(x_opt, xi))
        
        uncertainty_bounds = (np.percentile(values, 5), np.percentile(values, 95))
        
        result.uncertainty_bounds = uncertainty_bounds
        return result