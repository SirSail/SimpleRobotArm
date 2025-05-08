

# 🦾 Neural Network-Based 2D Robot Arm Controller

## 🧠 Description

This project implements a 2D robotic arm simulator controlled by a custom Multi-Layer Perceptron (MLP) neural network. It predicts joint angles (α and β) based on a desired end-effector position. The system includes a GUI for interactive control, training visualization, and error heatmaps for performance analysis. It's designed to demonstrate inverse kinematics using neural networks in a clear and engaging way.

---

## ⚙️ Installation Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/SirSail/SimpleRobotArm.git
   cd SimpleRobotArm
   ```

2. **Set up a virtual environment (optional but recommended)**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies**:

   ```bash
   pip install numpy matplotlib
   ```

4. **Run the application**:

   ```bash
   python robot_arm.py
   ```

---

## 🚀 Usage Instructions

* Once launched, adjust the **"Tip X"** and **"Tip Y"** sliders to move the robot arm's target position.
* The neural networks will predict the required joint angles in real time.
* The arm's movement and the joint angles will be updated dynamically in the GUI.

### Example:

```bash
python robot_arm.py
```

You will see console outputs like:

```
Target X: 0.97, Y: 0.95
Alpha: 0.85 rad, Beta: 1.04 rad
End Effector: x2=0.96, y2=0.94
------------------------------
```

---


##  License Information

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this software with attribution.

---

##  Contact Information

For questions, feedback, or contributions, feel free to reach out:

* GitHub: [@SirSail](https://github.com/SirSail)
* Email: [zeglinski.jakub@gmail.com](mailto:zeglinski.jakub@gmail.com)

---

