
### ⚙️ The Clinical Parameter Breakdown

To understand how Torgenesis steers the model, we must examine the internal pipeline: **Generate 3D Points ➔ Rotate ➔ Scale ➔ Project to High-Dimensional Latent Space ➔ Apply Weights ➔ Collapse to 1D Vector ➔ Inject.**

Here is the mechanical breakdown of exactly what each parameter does to the tensor math inside `toroidal_engine.py`.

#### 1. The Geometric Blueprint (Shape & Magnitude)
These parameters dictate the raw 3D coordinates generated before they ever touch the model's brain.

* **`major_radius` (R):** Sets the distance from the absolute center `(0,0,0)` to the middle of the torus "tube". 
    * *The Latent Effect:* Increasing this spreads the calculated points further away from the origin in 3D space. When projected into the model's 1536-dimensional space, a larger radius results in a "broader" steering vector with much wider variance (higher peaks and lower valleys).
* **`minor_radius` (r):** Sets the thickness of the tube itself.
    * *The Latent Effect:* If Major Radius is the "spread," Minor Radius is the "fuzziness" or "depth" of that spread. A large Minor Radius means the points on the inside of the donut ring are vastly different from the points on the outside, increasing the complexity of the final vector.
* **`scale`:** A literal scalar multiplier applied uniformly to all X, Y, Z coordinates.
    * *The Latent Effect:* It acts as a raw volume knob for the geometry before projection. A scale of `0.5` cuts the mathematical magnitude of the 3D shape in half, acting as a secondary intensity control.

#### 2. Resolution (The Tensor Shape)
These parameters dictate the exact size of the matrix being calculated.

* **`line_count_u` & `line_count_v`:** These define the grid resolution. `u` is the number of slices around the whole donut, and `v` is the number of slices around the tube.
    * *The Latent Effect:* Using `u=13` and `v=6` means the engine calculates exactly 78 discrete points in 3D space. When projected, this creates a `[78, 1536]` tensor. Your final steering vector is the collapsed sum of 78 distinct directional pushes. Increasing these numbers increases the "smoothness" of the sum but significantly raises VRAM compute overhead.

#### 3. Orientation & Motion (The Attention Shifters)
Because the projection matrix uses a fixed random seed, the angle at which the 3D shape hits that matrix completely changes the resulting vector.

* **`rotation_angles` (Pitch, Roll, Yaw):** Applies a standard 3D Euler rotation matrix to the 78 points. 
    * *The Latent Effect:* Dictates the "starting posture". Pitching the shape 90 degrees aligns completely different 3D points with the highest values of the projection matrix, shifting which specific concepts the vector initially triggers.
* **`rotation_velocity`:** Added to the base rotation angles and multiplied by the current token count (`step`).
    * *The Latent Effect:* Alters the vector on every single token, forcing the attention heads to continuously shift their focus.

##### The Mechanics of Velocity
The Injection is "Per-Token", Not "Per-Prompt." The engine recalculates the torus position for every single word generated. 
* **Low Velocity (e.g., 0.02):** The torus moves a fraction of an inch between tokens. The model's attention heads can easily adapt to this gentle, smooth trajectory, maintaining perfect grammar while smoothly altering its conceptual focus.
* **High Velocity (e.g., 2.5):** The torus spins violently between words. Token 1 gets shoved left, Token 2 gets shoved right, and Token 3 gets shoved upside down. This shakes the model's brain while it tries to write a sentence, destroying context continuity and shattering the output into broken syntax or hallucinations. 
* **The Takeaway:** Velocity is a "frequency of change" dial, not a "speed to destination". Keep it low for stability. 

#### 4. Mass & Gravity (The Collapse Modifiers)
The 78 projected points must be combined into a single 1D vector to be added to the model's hidden states.

* **`weighting_table`:** (e.g., `[0.10, 0.10, 0.10, 0.5...]`) The script tiles and repeats this sequence to cover all 78 points, multiplying each point by its corresponding weight before summing.
    * *The Latent Effect:* This creates 9 or 10 rhythmic, concentrated "bands" of heavy steering influence wrapping around the donut. The `0.5` spikes give certain points five times the mathematical mass. When collapsed, these heavy hotspots overwhelmingly dominate the final steering direction.
* **`center_weight`:** Projects the absolute origin `(0,0,0)` into the latent space and adds it to the summed torus vector.
    * *The Latent Effect:* The center point never moves. It acts as a static gravitational anchor. A `0.30` weight means 30% of your steering injection is a constant baseline, providing stability underneath the spinning geometry.

#### 5. The Void & The Master Fader
* **`void_intensity`:** Injects pure, uniform random noise (floats between -1.0 and 1.0) into the dimensions.
    * *The Latent Effect:* The Void Static acts as a latent disruptor. By vibrating the baseline hidden states, you prevent the model from locking into rigid attention pathways, making it highly susceptible to the directional push of your Toroid. 
* **`intensity_multiplier`:** The absolute master fader applied after the toroid is built, collapsed, and merged with the void.
    * *The Latent Effect:* A multiplier of `0.1` acts as a gentle suggestion. A multiplier of `2.0` is an absolute hijack, mathematically drowning out the native thoughts and forcing the model to pivot entirely to the Toroid's geometry.
    
Different models will be very active on different layers. Some models will have a "sweet spot" of a 0.4 multiplier, while others will want a multiplier of 1.4 to achieve desired results. The same goes for void static, if the model is outputting complete gibberish, dial up or down the values accordingly.    