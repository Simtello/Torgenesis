import customtkinter as ctk
import threading
import datetime
import torch
import os

# Pre-computation memory fix
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Import your existing engine classes
from steer_model import ModelSteerer
from toroidal_engine import ToroidalSteeringEngine

class TorgenesisGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("Torgenesis Control Center")
        self.geometry("1400x900") 
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Engine state variables
        self.steerer = None
        self.engine = None
        self.MODEL_ID = "local-safetensors-model-folder" 
        self.LATENT_DIM = None 

        # --- Layout Configuration (30/70 Split) ---
        self.grid_columnconfigure(0, weight=2) # Left panel (Tabs)
        self.grid_columnconfigure(1, weight=8) # Right panel (Console)
        self.grid_rowconfigure(0, weight=1)

        # --- Left Panel: Tabs ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Explicitly scale the tab buttons themselves
        self.tabview._segmented_button.configure(font=("Arial", 16, "bold"))
        
        self.tab_geo = self.tabview.add("Geometry & Scale")
        self.tab_motion = self.tabview.add("Motion & Rotation")
        self.tab_void = self.tabview.add("The Void & Injection")

        # Helper function for single-value rows (Scaled Up)
        def create_param_row(parent, label_text, desc_text, default_val, row):
            lbl = ctk.CTkLabel(parent, text=label_text, text_color="white", font=("Arial", 20, "bold"))
            lbl.grid(row=row, column=0, sticky="w", pady=(10, 0), padx=10)
            
            desc = ctk.CTkLabel(parent, text=desc_text, text_color="white", font=("Arial", 15, "italic"))
            desc.grid(row=row+1, column=0, sticky="w", pady=(0, 10), padx=10)
            
            ent = ctk.CTkEntry(parent, text_color="white", width=140, font=("Arial", 18))
            ent.insert(0, str(default_val))
            ent.grid(row=row, column=1, rowspan=2, sticky="e", padx=10)
            return ent

        # Helper function for multi-value rows (Scaled Up)
        def create_multi_param_row(parent, label_text, desc_text, default_vals, row):
            lbl = ctk.CTkLabel(parent, text=label_text, text_color="white", font=("Arial", 20, "bold"))
            lbl.grid(row=row, column=0, sticky="w", pady=(10, 0), padx=10)
            
            desc = ctk.CTkLabel(parent, text=desc_text, text_color="white", font=("Arial", 15, "italic"))
            desc.grid(row=row+1, column=0, sticky="w", pady=(0, 10), padx=10)
            
            # Sub-frame to hold the individual boxes
            box_frame = ctk.CTkFrame(parent, fg_color="transparent")
            box_frame.grid(row=row, column=1, rowspan=2, sticky="e", padx=10)
            
            entries = []
            for i, val in enumerate(default_vals):
                grid_row = i // 4
                grid_col = i % 4
                
                # Increased width to 75 to handle the larger font size
                ent = ctk.CTkEntry(box_frame, text_color="white", width=75, font=("Arial", 18))
                ent.insert(0, str(val))
                ent.grid(row=grid_row, column=grid_col, padx=3, pady=3)
                entries.append(ent)
            return entries

        # --- Tab 1: Geometry & Scale ---
        self.tab_geo.grid_columnconfigure(0, weight=1)
        self.ent_major = create_param_row(self.tab_geo, "Major Radius", "Primary outer ring size of the geometric trap.", "6.0", 0)
        self.ent_minor = create_param_row(self.tab_geo, "Minor Radius", "Thickness of the toroid's tube.", "4.0", 2)
        self.ent_u = create_param_row(self.tab_geo, "Line Count U", "Radial segments (density of the shape).", "13", 4)
        self.ent_v = create_param_row(self.tab_geo, "Line Count V", "Tubular segments.", "6", 6)
        self.ent_scale = create_param_row(self.tab_geo, "Scale", "Overall size multiplier within latent space.", "0.5", 8)
        self.ent_center = create_param_row(self.tab_geo, "Center Weight", "Gravitational pull of the absolute center point.", "0.30", 10)

        # --- Tab 2: Motion & Rotation ---
        self.tab_motion.grid_columnconfigure(0, weight=1)
        self.ent_rot_ang = create_multi_param_row(self.tab_motion, "Starting Angles (P, R, Y)", "Orients the shape before generation.", [0.01, 0.01, 0.01], 0)
        self.ent_rot_vel = create_multi_param_row(self.tab_motion, "Rotation Velocity (X, Y, Z)", "Spin speed per generated token.", [0.01, 0.01, 0.01], 2)

        # --- Tab 3: The Void & Injection ---
        self.tab_void.grid_columnconfigure(0, weight=1)
        self.ent_layer = create_param_row(self.tab_void, "Target Layer", "The specific neural network layer to intercept.", "14", 0)
        self.ent_mult = create_param_row(self.tab_void, "Intensity Multiplier", "Forcefully overrides the model's thoughts.", "0.6", 2)
        self.ent_void = create_param_row(self.tab_void, "Void Intensity", "Volume of the background 65,493-float static.", "0.14", 4)
        self.ent_tokens = create_param_row(self.tab_void, "Max Tokens", "Maximum length of the generated response.", "600", 6)
        self.ent_weights = create_multi_param_row(self.tab_void, "Weighting Table", "Distribution of structural mass.", [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], 8)

        # --- Right Panel: Live Console (Scaled Up) ---
        self.console_frame = ctk.CTkFrame(self)
        self.console_frame.grid(row=0, column=1, rowspan=2, padx=20, pady=20, sticky="nsew")
        self.console_frame.grid_rowconfigure(1, weight=1)
        self.console_frame.grid_columnconfigure(0, weight=1)
        
        lbl_console = ctk.CTkLabel(self.console_frame, text="Live Output Console", text_color="white", font=("Arial", 24, "bold"))
        lbl_console.grid(row=0, column=0, pady=10)
        
        self.txt_console = ctk.CTkTextbox(self.console_frame, text_color="white", font=("Arial", 26, "bold"), wrap="word")
        self.txt_console.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.log_to_console("System initialized. Standing by.")

        # --- Bottom Panel: Prompt & Controls (Scaled Up) ---
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.control_frame.grid_columnconfigure(0, weight=1)

        lbl_prompt = ctk.CTkLabel(self.control_frame, text="User Prompt", text_color="white", font=("Arial", 20, "bold"))
        lbl_prompt.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))

        self.txt_prompt = ctk.CTkTextbox(self.control_frame, text_color="white", height=100, font=("Arial", 18), wrap="word")
        self.txt_prompt.insert("1.0", "seed")
        self.txt_prompt.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        self.btn_seed = ctk.CTkButton(self.control_frame, text="Seed Prompt", text_color="white", font=("Arial", 20, "bold"), command=self.on_seed_click)
        self.btn_seed.grid(row=2, column=0, columnspan=2, pady=10)

    def log_to_console(self, text):
        self.txt_console.insert("end", text + "\n")
        self.txt_console.see("end")

    def clear_console(self):
        self.txt_console.delete("1.0", "end")

    def on_seed_click(self):
        self.btn_seed.configure(state="disabled")
        self.clear_console()
        threading.Thread(target=self.execute_injection, daemon=True).start()

    def execute_injection(self):
        try:
            # Parse all inputs from the GUI
            prompt = self.txt_prompt.get("1.0", "end").strip()
            target_layer = int(self.ent_layer.get())
            multiplier = float(self.ent_mult.get())
            max_tokens = int(self.ent_tokens.get())
            void_intensity = float(self.ent_void.get())

            # Parse the multi-box arrays
            engine_params = {
                'major_radius': float(self.ent_major.get()),
                'minor_radius': float(self.ent_minor.get()),
                'line_count_u': int(self.ent_u.get()),
                'line_count_v': int(self.ent_v.get()),
                'scale': float(self.ent_scale.get()),
                'rotation_angles': tuple(float(e.get()) for e in self.ent_rot_ang),
                'rotation_velocity': tuple(float(e.get()) for e in self.ent_rot_vel),
                'weighting_table': [float(e.get()) for e in self.ent_weights],
                'center_weight': float(self.ent_center.get())
            }

            # Boot the engine if it's the first run
            if self.steerer is None:
                self.log_to_console(f"Booting Neural Engine into VRAM... ({self.MODEL_ID})")
                self.log_to_console("This will take a moment on the first run.")
                self.steerer = ModelSteerer(model_id=self.MODEL_ID)
                
                # --- TRUE AUTOMATION FIX ---
                # Reach into the model's config and pull the exact hidden size dynamically
                self.LATENT_DIM = self.steerer.model.config.hidden_size
                self.log_to_console(f"Auto-detected Latent Dimension: {self.LATENT_DIM}")
                
                self.engine = ToroidalSteeringEngine(latent_dim=self.LATENT_DIM)
                self.log_to_console("Engine Online.\n")

            # Generate Void Static
            self.log_to_console(f"Generating {(self.LATENT_DIM * 32)} floats of Void Static...")
            void_pool = (torch.rand(self.LATENT_DIM * 32) * 2.0) - 1.0
            void_indices = torch.randint(0, len(void_pool), (self.LATENT_DIM,))
            void_noise_vector = void_pool[void_indices] * void_intensity
            empty_void = torch.zeros_like(void_noise_vector)

            # Run Baseline
            self.log_to_console(f"\n[EXECUTING BASELINE - LAYER {target_layer}]")
            torch.cuda.empty_cache()
            baseline_response = self.steerer.generate_with_steering(
                prompt=prompt, engine=self.engine, engine_params=engine_params,
                void_noise_vector=empty_void, layer_index=target_layer, 
                multiplier=0.0, max_new_tokens=max_tokens
            )
            self.log_to_console(baseline_response + "\n")

            # Run Steered Injection
            self.log_to_console(f"\n[EXECUTING STEERED - LAYER {target_layer} | MULT: {multiplier}]")
            torch.cuda.empty_cache()
            steered_response = self.steerer.generate_with_steering(
                prompt=prompt, engine=self.engine, engine_params=engine_params,
                void_noise_vector=void_noise_vector, layer_index=target_layer, 
                multiplier=multiplier, max_new_tokens=max_tokens
            )
            self.log_to_console(steered_response + "\n")

            # Continuous Append Logging
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"""
======================================================
[TIMESTAMP: {timestamp}]
[PROMPT]: {prompt}
[DIALS]: Layer {target_layer} | Multiplier {multiplier} | Void {void_intensity} | Scale {engine_params['scale']}
[MOTION]: Vel {engine_params['rotation_velocity']} | Start {engine_params['rotation_angles']}
------------------------------------------------------
[BASELINE OUTPUT]
{baseline_response}

[STEERED OUTPUT]
{steered_response}
======================================================\n"""
            
            with open("torgenesis_log.txt", "a", encoding="utf-8") as f:
                f.write(log_entry)
                
            self.log_to_console(">>> Experiment safely appended to torgenesis_log.txt <<<")

        except Exception as e:
            self.log_to_console(f"\n[ERROR]: {str(e)}")
            
        finally:
            self.btn_seed.configure(state="normal")

if __name__ == "__main__":
    app = TorgenesisGUI()
    app.mainloop()