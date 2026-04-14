
# Torgenesis: Dynamic Toroidal Activation Steering

Torgenesis is an experimental framework for real-time, geometric activation steering in Large Language Models. By utilizing PyTorch forward hooks and toroidal mathematics, this engine allows developers to dynamically inject shaped, rotating mathematical structures into a model's hidden states during the generation process.

Optimized for local AI environments, the framework leverages 4-bit NF4 quantization to efficiently compress and manage VRAM, allowing developers to scale their experiments across any model size or hardware configuration, making it ideal for robust local development and testing.

## 🧠 Core Mechanics

Torgenesis fundamentally changes how we approach activation steering by moving away from static vector additions and introducing **geometric, time-based vector modulation**.

### 1. The Toroidal Mathematics
Instead of simply pushing a model's activations toward a static concept vector, Torgenesis constructs a multi-dimensional Torus within the model's latent space. The geometry is generated using the following parametric equations:

$$x = (R + r \cos(\phi)) \cos(\theta)$$
$$y = (R + r \cos(\phi)) \sin(\theta)$$
$$z = r \sin(\phi)$$

Where:
* $R$ is the **Major Radius** (the primary outer ring).
* $r$ is the **Minor Radius** (the thickness of the tube).
* $\theta$ and $\phi$ are angles defining the matrix grid (controlled by `line_count_u` and `line_count_v`).

**Latent Projection:**
Because LLMs operate in high-dimensional space (e.g., 2048 dimensions), the 3D toroidal coordinates cannot be injected directly. Torgenesis multiplies the 3D points against a fixed-seed, pseudo-random projection matrix. This creates a stable, consistent mapping from 3D space into the LLM's $N$-dimensional latent space, ensuring the geometry holds its "shape" during injection.

### 2. Time-Based Rotation
The Toroid is not static. By applying a 3D rotation matrix based on continuous velocity, the shape continuously spins. As the model generates each new token, the internal clock ($step$) advances:

$$\text{Rotation}_{current} = \text{Base Angle} + (\text{Velocity} \times step)$$

This recalculates the Toroid's position, exposing the model to a dynamically shifting geometric influence for every single token it produces.

### 3. Surgical Intervention via PyTorch Hooks
Torgenesis utilizes PyTorch's `register_forward_hook` to perform non-destructive surgery on the model's neural layers in real-time. 

Here is the step-by-step lifecycle of an injection:
1.  **Targeting:** A specific transformer layer is selected for interception.
2.  **The Void Static:** A massive pool of static noise is generated. The dynamically calculated Toroid vector is submerged into this static. 
3.  **The Hook:** As the forward pass reaches the target layer, the hook halts the process. It calculates the precise structural vector for that exact millisecond of generation.
4.  **Injection:** The final vector is format-matched to the model's precision (16-bit), scaled by an Intensity Multiplier, and fused directly into the hidden states.
5.  **Release:** The modified hidden states are returned to the model, forcing it to generate the next token through the lens of the spinning Toroid. 

## 🎛️ Comprehensive Parameter Guide
The framework offers an intricate level of control over the mathematical structure being injected. Every parameter can be adjusted in real-time via the Torgenesis Control Center.

**Geometry & Scale**
* **Major Radius:** Defines the primary outer ring size of the geometric net.
* **Minor Radius:** Dictates the thickness of the toroid's tube.
* **Line Count U & V:** Controls the radial and tubular segment density, acting as the "resolution" of the shape.
* **Scale:** Adjusts the overall size multiplier of the structure within the latent space.
* **Center Weight:** Modifies the gravitational pull of the absolute center point of the toroid.

**Motion & Rotation**
* **Starting Angles (Pitch, Roll, Yaw):** Orients the initial 3D position of the shape before generation begins.
* **Rotation Velocity (X, Y, Z):** Defines the continuous spin speed per generated token, creating a dynamic, moving net.

**The Void & Injection**
* **Target Layer:** Specifies the exact neural network layer (e.g., Layer 14) to intercept and modify during the forward pass.
* **Intensity Multiplier:** The leveraged scaling factor that overrides the model's baseline thoughts with the generated geometry.
* **Void Intensity:** Controls the volume/loudness of the background float static the shape is submerged in.
* **Weighting Table:** An array that distributes structural mass unevenly across the toroid, creating dense or hollow sections.
* **Max Tokens:** Sets the maximum length of the generated response.

## 🛠️ Features

* **CustomTkinter Control Center:** A dark-themed, 30/70 split GUI for manipulating geometric parameters, rotational velocity, and void injection in real-time without modifying code.
* **Live Output Console:** Real-time side-by-side execution logging of Baseline (unsteered) vs. Steered generations.
* **VRAM Efficient:** Built-in `BitsAndBytesConfig` ensures maximum local efficiency, utilizing double quantization and `float16` compute data types.
* **Continuous Logging:** Auto-appends experimental configurations, seeds, and outputs directly to `torgenesis_log.txt` for chronological tracking.

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* PyTorch (CUDA-enabled)
* Transformers, Accelerate, BitsAndBytes
* CustomTkinter
* * **Model Format:** The engine supports standard Hugging Face architectures (`.safetensors`) as well as native `.gguf` files, thanks to recent updates in the `transformers` library.

### Installation
1. Clone the repository:
```bash
git clone https://github.com/simtello/torgenesis.git
cd torgenesis
```
2. Install the required dependencies:
```bash
pip install torch torchvision torchaudio transformers accelerate bitsandbytes customtkinter
```
3. Ensure your local model weights (e.g., Llama, Gemma) are correctly pathed in the GUI script or loaded within your local directory.

### Usage
Launch the Control Center to begin experimenting with latent geometries:
```bash
python torgenesis_gui.py
```
Adjust your Major/Minor radiuses, set your target layer, and click **Seed Prompt** to watch the model navigate the injected architecture. 

## ⚠️ Notes on Memory Management
The GUI applies `PYTORCH_ALLOC_CONF="expandable_segments:True"` to mitigate memory fragmentation during heavy, continuous prompt seeding. The framework also actively forces `torch.cuda.empty_cache()` between baseline and steered runs to ensure stability across extended testing sessions.
