# === Imports ===
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sys

# === Constants ===
SEGMENT_LENGTH = 1.0  # Length of each arm segment

# === Forward Kinematics ===
def forward_kinematics(angle_alpha, angle_beta):
    x = SEGMENT_LENGTH * np.cos(angle_alpha) + SEGMENT_LENGTH * np.cos(angle_alpha + angle_beta)
    y = SEGMENT_LENGTH * np.sin(angle_alpha) + SEGMENT_LENGTH * np.sin(angle_alpha + angle_beta)
    return np.array([x, y])

# === Training Data Generation ===
def generate_training_data(samples=1000):
    inputs, targets = [], []
    for _ in range(samples):
        angle_alpha = np.random.uniform(0.01, np.pi - 0.01)
        angle_beta = np.random.uniform(0.01, np.pi - 0.01)
        position = forward_kinematics(angle_alpha, angle_beta) / 2.0
        norm_alpha = 0.8 * (angle_alpha / np.pi) + 0.1
        norm_beta = 0.8 * (angle_beta / np.pi) + 0.1
        inputs.append(position)
        targets.append([norm_alpha, norm_beta])

    inputs_np = np.array(inputs)
    plt.figure(figsize=(6, 6))
    plt.scatter(inputs_np[:, 0], inputs_np[:, 1], s=2)
    plt.title("Training Data Distribution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    return np.array(inputs), np.array(targets)

# === MLP Neural Network ===
class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.005):
        self.layers = []
        self.learning_rate = learning_rate
        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            biases = np.zeros(layer_sizes[i + 1])
            self.layers.append({'weights': weights, 'bias': biases})

    def activation_sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def activation_sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def activation_relu(self, x):
        return np.maximum(0, x)

    def activation_relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        for i, layer in enumerate(self.layers):
            z = np.dot(self.activations[-1], layer['weights']) + layer['bias']
            self.z_values.append(z)
            if i < len(self.layers) - 1:
                a = self.activation_relu(z)
            else:
                a = self.activation_sigmoid(z)
            self.activations.append(a)
        return self.activations[-1]

    def backward(self, X, y):
        output = self.activations[-1]
        error = y - output
        deltas = [error * self.activation_sigmoid_derivative(output)]

        for i in reversed(range(len(self.layers) - 1)):
            delta = np.dot(deltas[-1], self.layers[i + 1]['weights'].T)
            delta *= self.activation_relu_derivative(self.z_values[i])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.layers)):
            dw = np.dot(self.activations[i].T, deltas[i])
            db = np.sum(deltas[i], axis=0)
            self.layers[i]['weights'] += self.learning_rate * dw
            self.layers[i]['bias'] += self.learning_rate * db

    def train(self, X, y, epochs=10):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean((y - output) ** 2)
            self.backward(X, y)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

# === GUI Application ===
class ArmApp:
    def __init__(self, root, mlp_alpha, mlp_beta):
        self.root = root
        self.mlp_alpha = mlp_alpha
        self.mlp_beta = mlp_beta

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_xlim(-2.2 * SEGMENT_LENGTH, 2.2 * SEGMENT_LENGTH)
        self.ax.set_ylim(-2.2 * SEGMENT_LENGTH, 2.2 * SEGMENT_LENGTH)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.line1, = self.ax.plot([], [], 'r-', linewidth=4)
        self.line2, = self.ax.plot([], [], 'b-', linewidth=4)
        self.joints, = self.ax.plot([], [], 'ko', markersize=8)
        self.tip, = self.ax.plot([], [], 'go', markersize=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.x_slider = tk.Scale(root, from_=-1, to=1, resolution=0.01, orient="horizontal", label="Tip X", command=self.update_arm)
        self.x_slider.set(1.0)
        self.x_slider.pack(fill="x")

        self.y_slider = tk.Scale(root, from_=-1, to=1, resolution=0.01, orient="horizontal", label="Tip Y", command=self.update_arm)
        self.y_slider.set(1.0)
        self.y_slider.pack(fill="x")

        self.update_arm(None)

    def update_arm(self, _):
        try:
            x = self.x_slider.get()
            y = self.y_slider.get()
            target_position = np.array([[x, y]]) / 2.0

            angle_alpha = ((self.mlp_alpha.forward(target_position)).item() - 0.1) / 0.8 * np.pi
            angle_beta = ((self.mlp_beta.forward(target_position)).item() - 0.1) / 0.8 * np.pi

            angle_alpha = np.clip(angle_alpha, 0.01, np.pi - 0.01)
            angle_beta = np.clip(angle_beta, 0.01, np.pi - 0.01)

            x0, y0 = 0, 0
            x1 = SEGMENT_LENGTH * np.cos(angle_alpha)
            y1 = SEGMENT_LENGTH * np.sin(angle_alpha)
            x2 = x1 + SEGMENT_LENGTH * np.cos(angle_alpha + angle_beta)
            y2 = y1 + SEGMENT_LENGTH * np.sin(angle_alpha + angle_beta)

            print(f"Target X: {x:.2f}, Y: {y:.2f}")
            print(f"Alpha: {angle_alpha:.2f} rad, Beta: {angle_beta:.2f} rad")
            print(f"End Effector: x2={x2:.2f}, y2={y2:.2f}")
            print("-" * 30)

            self.line1.set_data([x0, x1], [y0, y1])
            self.line2.set_data([x1, x2], [y1, y2])
            self.joints.set_data([x0, x1, x2], [y0, y1, y2])
            self.tip.set_data([x2], [y2])
            self.ax.set_title(f"Alpha={angle_alpha:.2f}, Beta={angle_beta:.2f}", fontsize=12)
            self.canvas.draw()
        except Exception as e:
            print("Error in update_arm:", e, file=sys.stderr)

# === Heatmap of Prediction Error ===
def plot_error_heatmap(mlp_alpha, mlp_beta, grid_size=50):
    xs = np.linspace(-1, 1, grid_size)
    ys = np.linspace(-1, 1, grid_size)
    error_grid = np.zeros((grid_size, grid_size))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            target_position = np.array([[x, y]]) / 2.0
            alpha_pred = mlp_alpha.forward(target_position).item()
            beta_pred = mlp_beta.forward(target_position).item()

            angle_alpha = np.clip((alpha_pred - 0.1) / 0.8 * np.pi, 0.01, np.pi - 0.01)
            angle_beta = np.clip((beta_pred - 0.1) / 0.8 * np.pi, 0.01, np.pi - 0.01)

            x1 = SEGMENT_LENGTH * np.cos(angle_alpha)
            y1 = SEGMENT_LENGTH * np.sin(angle_alpha)
            x2 = x1 + SEGMENT_LENGTH * np.cos(angle_alpha + angle_beta)
            y2 = y1 + SEGMENT_LENGTH * np.sin(angle_alpha + angle_beta)

            error = np.sqrt((x2 - x)**2 + (y2 - y)**2)
            error_grid[j, i] = error

    plt.figure(figsize=(6, 5))
    plt.imshow(error_grid, extent=(-1, 1, -1, 1), origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar(label='End Effector Error (m)')
    plt.title("Prediction Error Heatmap")
    plt.xlabel("Target X")
    plt.ylabel("Target Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Main Execution ===
def main():
    print("Generating training data...")
    inputs, targets = generate_training_data(samples=1000)

    print("Training MLPs...")
    mlp_alpha = MLP(input_size=2, hidden_layers=[30, 20, 10], output_size=1)
    mlp_beta = MLP(input_size=2, hidden_layers=[30, 20, 10], output_size=1)

    mlp_alpha.train(inputs, targets[:, 0].reshape(-1, 1), epochs=10000)
    mlp_beta.train(inputs, targets[:, 1].reshape(-1, 1), epochs=10000)

    plot_error_heatmap(mlp_alpha, mlp_beta)

    print("Launching GUI...")
    root = tk.Tk()
    root.title("Robot Arm Controller")

    def on_closing():
        root.destroy()
        sys.exit()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = ArmApp(root, mlp_alpha, mlp_beta)
    root.mainloop()

if __name__ == "__main__":
    main()