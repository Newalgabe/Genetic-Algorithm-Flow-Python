# Genetic-Algorithm-Flow-Python

This tool helps you visualize and optimize a function using evolutionary algorithms. Tune parameters and watch real-time progress as the algorithm searches for the optimal solution.

---

## Features

- **Function Input**: 
  Enter your optimization function (e.g., `-1*x[0]**2-100`) in the **Function f(x)** field.

- **Parameters**: 
  Adjust population size, alpha, mutation rate, and more through easy-to-use sliders and input fields.

- **Real-time Results**: 
  Watch the algorithm improve solutions in real-time as it finds better results with each iteration.

- **Progress Plot**: 
  Visual graph showing fitness improvement over time, allowing you to track the progress of the optimization.

- **Status Updates**: 
  View the current best solution and fitness values as the algorithm runs.

---

## How to Use

1. **Enter your function**: 
   Type your optimization function into the **Function f(x)** field.

2. **Adjust parameters**: 
   Use the sliders and input fields to modify key parameters like population size, alpha, and mutation rate.

3. **Start optimization**: 
   Click on the **🚀 Start Optimization** button to begin the process.

4. **Monitor progress**: 
   Switch to the **📊 Results** tab to see the live updates and progress of the optimization.

5. **Stop or Reset**: 
   If needed, you can stop the process using the **⏹️ Stop** button or reset everything with the **🔄 Reset** button.

---

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/optimization-algorithm.git
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python app.py
    ```

---

## Example

For an example, you can try optimizing the following function:

```python
def f(x):
    return -1 * x[0]**2 - 100
