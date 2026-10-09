# Neuro Racer

A retro-style top-down arcade racing game built from scratch in Python/Pygame, featuring a custom 2D physics engine, procedurally generated tracks, and a self-learning Neuroevolution AI opponent. 

## Features
- **Neuroevolution AI**: The AI driving logic isn't hardcoded. It uses raycasting vision to "see" the track boundaries and learns how to drive, drift, and navigate via a custom evolutionary neural network.
- **Procedural Track Generation**: Infinite, random, smoothly-curved racetracks generated mathematically.
- **Custom Arcade Physics**: High-speed, high-grip mechanics tailored to feel like classic 90s top-down racers.
- **Local Multiplayer**: Race head-to-head with a friend on the same keyboard, alongside the trained AI.
- **Retro Aesthetic**: A sleek arcade aesthetic with a dynamic zoom camera, satisfying tire trails, and a soothing parallax starfield menu.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neuro-racer.git
   cd neuro-racer
   ```
2. Install the required dependencies (Pygame & NumPy):
   ```bash
   pip install -r requirements.txt
   ```

## How to Play

### Race (Manual Drive)
To jump straight into the game and race against a friend or the pre-trained AI:
```bash
python manual_drive.py
```
**Controls:**
- **Player 1**: `Arrow Keys` (Up to accelerate, Left/Right to steer)
- **Player 2**: `W A S D` (W to accelerate, A/D to steer)
- **Menu/Pause**: `ESC`

### Train the AI
If you want to train your own AI model from scratch, you can run the training visualization:
```bash
python main.py
```
- **Training Controls**: Press `SPACE` to pause, `F` for fast-forward, `+/-` to change simulation speed, and `S` to safely save a snapshot of the current best brain to the `models/` directory.
- **Headless Mode**: For maximum training speed without rendering, run `python main.py --headless 500`.
