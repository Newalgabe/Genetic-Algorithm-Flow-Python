import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import random
from math import *
import itertools

class GeneticAlgorithmGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Genetic Algorithm Optimizer")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.function_var = tk.StringVar(value="-1*x[0]**2-100")
        self.population_var = tk.IntVar(value=100)
        self.alpha_var = tk.DoubleVar(value=0.5)
        self.deviation_var = tk.DoubleVar(value=2.5)
        self.mutation_rate_var = tk.DoubleVar(value=0.0001)
        self.iterations_var = tk.IntVar(value=100000)
        
        # Algorithm state
        self.is_running = False
        self.current_iteration = 0
        self.best_fitness_history = []
        self.best_solution_history = []
        self.last_update_iteration = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container with padding
        main_frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="🧬 Genetic Algorithm Optimizer", 
                              font=('Arial', 24, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Configuration tab
        config_frame = tk.Frame(notebook, bg='#f0f0f0')
        notebook.add(config_frame, text="⚙️ Configuration")
        
        # Results tab
        results_frame = tk.Frame(notebook, bg='#f0f0f0')
        notebook.add(results_frame, text="📊 Results")
        
        self.setup_config_tab(config_frame)
        self.setup_results_tab(results_frame)
        
    def setup_config_tab(self, parent):
        # Function input section
        func_frame = tk.LabelFrame(parent, text="🎯 Objective Function", 
                                  font=('Arial', 12, 'bold'), bg='#f0f0f0', 
                                  fg='#2c3e50', padx=10, pady=10)
        func_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(func_frame, text="Function f(x):", font=('Arial', 10), 
                bg='#f0f0f0').pack(anchor='w')
        
        func_entry = tk.Entry(func_frame, textvariable=self.function_var, 
                             font=('Arial', 11), width=60)
        func_entry.pack(fill=tk.X, pady=5)
        
        # Examples
        examples_label = tk.Label(func_frame, 
                                text="Examples: -1*x[0]**2-100, x[0]**2+x[1]**2, sin(x[0])*cos(x[1])", 
                                font=('Arial', 9), bg='#f0f0f0', fg='#7f8c8d')
        examples_label.pack(anchor='w')
        
        # Parameters section
        params_frame = tk.LabelFrame(parent, text="🔧 Algorithm Parameters", 
                                   font=('Arial', 12, 'bold'), bg='#f0f0f0', 
                                   fg='#2c3e50', padx=10, pady=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Population size
        pop_frame = tk.Frame(params_frame, bg='#f0f0f0')
        pop_frame.pack(fill=tk.X, pady=5)
        tk.Label(pop_frame, text="Population Size:", font=('Arial', 10), 
                bg='#f0f0f0').pack(side=tk.LEFT)
        pop_scale = tk.Scale(pop_frame, from_=50, to=500, orient=tk.HORIZONTAL, 
                           variable=self.population_var, length=300)
        pop_scale.pack(side=tk.RIGHT)
        
        # Alpha parameter
        alpha_frame = tk.Frame(params_frame, bg='#f0f0f0')
        alpha_frame.pack(fill=tk.X, pady=5)
        tk.Label(alpha_frame, text="Alpha (Crossover):", font=('Arial', 10), 
                bg='#f0f0f0').pack(side=tk.LEFT)
        alpha_entry = tk.Entry(alpha_frame, textvariable=self.alpha_var, 
                              font=('Arial', 10), width=10)
        alpha_entry.pack(side=tk.RIGHT)
        
        # Deviation parameter
        dev_frame = tk.Frame(params_frame, bg='#f0f0f0')
        dev_frame.pack(fill=tk.X, pady=5)
        tk.Label(dev_frame, text="Mutation Deviation:", font=('Arial', 10), 
                bg='#f0f0f0').pack(side=tk.LEFT)
        dev_entry = tk.Entry(dev_frame, textvariable=self.deviation_var, 
                            font=('Arial', 10), width=10)
        dev_entry.pack(side=tk.RIGHT)
        
        # Mutation rate
        mut_frame = tk.Frame(params_frame, bg='#f0f0f0')
        mut_frame.pack(fill=tk.X, pady=5)
        tk.Label(mut_frame, text="Mutation Rate:", font=('Arial', 10), 
                bg='#f0f0f0').pack(side=tk.LEFT)
        mut_entry = tk.Entry(mut_frame, textvariable=self.mutation_rate_var, 
                            font=('Arial', 10), width=10)
        mut_entry.pack(side=tk.RIGHT)
        
        # Iterations
        iter_frame = tk.Frame(params_frame, bg='#f0f0f0')
        iter_frame.pack(fill=tk.X, pady=5)
        tk.Label(iter_frame, text="Max Iterations:", font=('Arial', 10), 
                bg='#f0f0f0').pack(side=tk.LEFT)
        iter_scale = tk.Scale(iter_frame, from_=1000, to=1000000, 
                            orient=tk.HORIZONTAL, variable=self.iterations_var, 
                            length=300, tickinterval=100000)
        iter_scale.pack(side=tk.RIGHT)
        
        # Control buttons
        button_frame = tk.Frame(parent, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, padx=10, pady=20)
        
        self.start_button = tk.Button(button_frame, text="🚀 Start Optimization", 
                                    command=self.start_optimization, 
                                    font=('Arial', 12, 'bold'), 
                                    bg='#27ae60', fg='white', 
                                    padx=20, pady=10)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="⏹️ Stop", 
                                   command=self.stop_optimization, 
                                   font=('Arial', 12, 'bold'), 
                                   bg='#e74c3c', fg='white', 
                                   padx=20, pady=10, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(button_frame, text="🔄 Reset", 
                                    command=self.reset_optimization, 
                                    font=('Arial', 12, 'bold'), 
                                    bg='#f39c12', fg='white', 
                                    padx=20, pady=10)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
    def setup_results_tab(self, parent):
        # Results display
        results_frame = tk.Frame(parent, bg='#f0f0f0')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current status
        status_frame = tk.LabelFrame(results_frame, text="📈 Current Status", 
                                   font=('Arial', 12, 'bold'), bg='#f0f0f0', 
                                   fg='#2c3e50', padx=10, pady=10)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_text = tk.Text(status_frame, height=6, font=('Arial', 10), 
                                 bg='#ecf0f1', fg='#2c3e50')
        self.status_text.pack(fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Plot frame
        plot_frame = tk.LabelFrame(results_frame, text="📊 Optimization Progress", 
                                 font=('Arial', 12, 'bold'), bg='#f0f0f0', 
                                 fg='#2c3e50', padx=10, pady=10)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4), facecolor='white')
        self.ax.set_title('Fitness Progress', fontsize=12, fontweight='bold')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Best Fitness')
        self.ax.grid(True, alpha=0.3)
        
        # Embed plot in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def start_optimization(self):
        if not self.validate_inputs():
            return
            
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Clear previous results
        self.best_fitness_history = []
        self.best_solution_history = []
        self.current_iteration = 0
        
        # Start optimization in separate thread
        self.optimization_thread = threading.Thread(target=self.run_optimization)
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        
    def stop_optimization(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
    def reset_optimization(self):
        self.stop_optimization()
        self.best_fitness_history = []
        self.best_solution_history = []
        self.current_iteration = 0
        self.last_update_iteration = 0
        self.progress_var.set(0)
        if hasattr(self, 'status_text') and self.status_text.winfo_exists():
            self.status_text.delete(1.0, tk.END)
        if hasattr(self, 'ax') and self.ax:
            self.ax.clear()
            self.ax.set_title('Fitness Progress', fontsize=12, fontweight='bold')
            self.ax.set_xlabel('Iteration')
            self.ax.set_ylabel('Best Fitness')
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
        
    def validate_inputs(self):
        try:
            # Test function
            test_x = [1.0] * self.get_dimension()
            eval(self.function_var.get().replace('x[', 'test_x['))
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Invalid function: {str(e)}")
            return False
            
    def get_dimension(self):
        func = self.function_var.get()
        max_dim = 0
        for i in range(10):
            if f'x[{i}]' in func:
                max_dim = i + 1
        return max_dim if max_dim > 0 else 1
        
    def run_optimization(self):
        # Get parameters
        function = self.function_var.get()
        pop_size = self.population_var.get()
        alpha = self.alpha_var.get()
        deviation = self.deviation_var.get()
        mutation_rate = self.mutation_rate_var.get()
        max_iterations = self.iterations_var.get()
        dimension = self.get_dimension()
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            individual = [random.uniform(-100, 100) for _ in range(dimension)]
            population.append(individual)
            
        best_fitness = float('-inf')
        best_solution = None
        
        for iteration in range(max_iterations):
            if not self.is_running:
                break
                
            self.current_iteration = iteration
            
            # Evaluate fitness
            fitness_values = []
            for individual in population:
                try:
                    x = individual
                    fitness = eval(function.replace('x[', 'x['))
                    fitness_values.append(fitness)
                except:
                    fitness_values.append(float('-inf'))
            
            # Find best individual
            max_fitness = max(fitness_values)
            best_idx = fitness_values.index(max_fitness)
            
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_solution = population[best_idx].copy()
                self.best_fitness_history.append(max_fitness)
                self.best_solution_history.append(best_solution.copy())
            
            # Update GUI safely (limit frequency)
            if iteration - self.last_update_iteration >= max(1, max_iterations // 100):
                try:
                    self.root.after(0, self.update_gui, iteration, max_iterations, 
                                  best_solution, best_fitness)
                    self.last_update_iteration = iteration
                except Exception as e:
                    print(f"GUI scheduling error: {e}")
            
            # Selection, crossover, mutation
            population = self.genetic_operations(population, fitness_values, 
                                              alpha, deviation, mutation_rate)
            
        self.root.after(0, self.optimization_complete)
        
    def genetic_operations(self, population, fitness_values, alpha, deviation, mutation_rate):
        # Selection (tournament selection)
        new_population = []
        for _ in range(len(population)):
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_values[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            new_population.append(population[winner_idx].copy())
        
        # Crossover
        offspring = []
        for i in range(0, len(new_population), 2):
            if i + 1 < len(new_population):
                parent1 = np.array(new_population[i])
                parent2 = np.array(new_population[i + 1])
                
                # Differential crossover
                child1 = parent1 + alpha * (parent2 - parent1)
                child2 = parent2 + alpha * (parent1 - parent2)
                
                offspring.extend([child1.tolist(), child2.tolist()])
            else:
                offspring.append(new_population[i])
        
        # Mutation
        for individual in offspring:
            for i in range(len(individual)):
                if random.random() < mutation_rate:
                    individual[i] += random.gauss(0, deviation)
                    individual[i] = max(-1000, min(1000, individual[i]))  # Clamp values
        
        return offspring[:len(population)]
        
    def update_gui(self, iteration, max_iterations, best_solution, best_fitness):
        try:
            # Update progress
            progress = (iteration / max_iterations) * 100
            self.progress_var.set(progress)
            
            # Update status text safely
            if hasattr(self, 'status_text') and self.status_text.winfo_exists():
                self.status_text.delete(1.0, tk.END)
                status = f"Iteration: {iteration}/{max_iterations}\n"
                status += f"Best Fitness: {best_fitness:.6f}\n"
                status += f"Best Solution: {[round(x, 4) for x in best_solution]}\n"
                status += f"Progress: {progress:.1f}%"
                self.status_text.insert(tk.END, status)
            
            # Update plot safely
            if len(self.best_fitness_history) > 1 and hasattr(self, 'ax') and self.ax:
                try:
                    self.ax.clear()
                    self.ax.plot(self.best_fitness_history, 'b-', linewidth=2)
                    self.ax.set_title('Fitness Progress', fontsize=12, fontweight='bold')
                    self.ax.set_xlabel('Iteration')
                    self.ax.set_ylabel('Best Fitness')
                    self.ax.grid(True, alpha=0.3)
                    self.canvas.draw()
                except Exception as e:
                    print(f"Plot update error: {e}")
        except Exception as e:
            print(f"GUI update error: {e}")
        
    def optimization_complete(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        if self.best_solution_history:
            final_solution = self.best_solution_history[-1]
            final_fitness = self.best_fitness_history[-1]
            
            messagebox.showinfo("Optimization Complete", 
                              f"Final Best Fitness: {final_fitness:.6f}\n"
                              f"Final Best Solution: {[round(x, 4) for x in final_solution]}")
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = GeneticAlgorithmGUI()
    app.run()
