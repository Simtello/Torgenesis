import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from toroidal_engine import ToroidalSteeringEngine

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from toroidal_engine import ToroidalSteeringEngine

class ModelSteerer:
    def __init__(self, model_id, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Loading tokenizer from {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        
        # --- ADDED: 4-Bit NF4 Quantization ---
        # Compresses 7B/8B models to fit perfectly inside 12GB VRAM
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        print(f"Loading model from {model_id} in 4-bit precision...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16, 
            trust_remote_code=True, # --- ADDED THIS LINE ---
            local_files_only=True
        )
        self.hook_handle = None
        self.current_step = 0 # Internal clock for dynamic rotation

    def _create_injection_hook(self, engine, engine_params, multiplier, void_noise_vector):
        """
        The surgical hook now recalculates the Toroid on the fly and merges it 
        with the Void Static for every single token.
        """
        def hook(module, input, output):
            # 1. Generate the Toroid vector for the current point in time (rotation)
            base_steering_vector = engine.generate_steering_vector(engine_params, step=self.current_step)
            
            # 2. Submerge the Toroid in the Void Static
            final_vector = base_steering_vector + void_noise_vector.to(base_steering_vector.device)
            
            # 3. Format the injection to match the model's brain (16-bit)
            if isinstance(output, tuple):
                hidden_states = output[0]
                injection = (final_vector.to(hidden_states.device).to(hidden_states.dtype) * multiplier)
                modified_states = hidden_states + injection
                self.current_step += 1 # Advance the clock for the next token
                return (modified_states,) + output[1:]
            else:
                hidden_states = output
                injection = (final_vector.to(hidden_states.device).to(hidden_states.dtype) * multiplier)
                modified_states = hidden_states + injection
                self.current_step += 1 # Advance the clock for the next token
                return modified_states
                
        return hook

    def generate_with_steering(self, prompt, engine, engine_params, void_noise_vector, layer_index, multiplier, max_new_tokens=150):
        # Reset the clock for a fresh generation
        self.current_step = 0 
        
        target_layer = self.model.model.layers[layer_index]
        self.hook_handle = target_layer.register_forward_hook(
            self._create_injection_hook(engine, engine_params, multiplier, void_noise_vector)
        )
        
        print(f"\n[Hook Attached to Layer {layer_index} | Multiplier: {multiplier} | Motion Active]")
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        print("Generating response...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=50,       # Force it to keep talking in the void
                repetition_penalty=1.15, # Prevent looping breakdown
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        self.hook_handle.remove()
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start = full_text.find("assistant\n")
        if response_start != -1:
            return full_text[response_start + 10:].strip()
        return full_text.strip()


# ==========================================
# PARAMETER ADJUSTMENT AND EXECUTION
# ==========================================
if __name__ == "__main__":
    
    MODEL_ID = "local-safetensors-model-folder/" 
    LATENT_DIM = 2048 
    
    engine_params = {
        'major_radius': 6.0,
        'minor_radius': 4.0,
        'line_count_u': 13,
        'line_count_v': 6,
        'scale': 0.5, 
        
        # Base starting angles (Pitch, Roll, Yaw)
        'rotation_angles': (0.10, 0.10, 0.10), 
        
        # --- NEW: MOTION CONTROLS ---
        # Set all to 0.0 for a STATIC toroid. 
        # Add values to make it SPIN per token.
        'rotation_velocity': (0.10, 0.10, 0.10), 
        
        'weighting_table': [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10], 
        'center_weight': 0.30 
    }
    
    TARGET_LAYER = 14          
    INTENSITY_MULTIPLIER = 0.4
    MAX_TOKENS = 300           
    VOID_INTENSITY = 0.15 # How loud the static is (10%)
    
    USER_PROMPT = "...seed..."

    # ==========================================
    # --- THE VOID STATIC GENERATOR ---
    print("Generating 65,493 floats of Void Static...")
    void_pool = (torch.rand(65493) * 2.0) - 1.0
    void_indices = torch.randint(0, 65493, (LATENT_DIM,))
    void_noise_vector = void_pool[void_indices] * VOID_INTENSITY
    
    # Generate an empty void (zeros) for the baseline
    empty_void = torch.zeros_like(void_noise_vector)

    print("Initializing Dynamic Toroidal Engine...")
    engine = ToroidalSteeringEngine(latent_dim=LATENT_DIM) 
    
    steerer = ModelSteerer(model_id=MODEL_ID)
    
    # --- RESTORED: Generate the Baseline Response ---
    print("\n================ BASELINE (NO STEERING) ================")
    baseline_response = steerer.generate_with_steering(
        prompt=USER_PROMPT, 
        engine=engine,
        engine_params=engine_params,
        void_noise_vector=empty_void, 
        layer_index=TARGET_LAYER, 
        multiplier=0.0,
        max_new_tokens=MAX_TOKENS
    )
    
    # Print baseline to screen
    print("\n[BASELINE OUTPUT]")
    print(baseline_response)
    
    torch.cuda.empty_cache()
    
    print("\n================ STEERED (TOROID) ================")
    steered_response = steerer.generate_with_steering(
        prompt=USER_PROMPT, 
        engine=engine,
        engine_params=engine_params,
        void_noise_vector=void_noise_vector,
        layer_index=TARGET_LAYER, 
        multiplier=INTENSITY_MULTIPLIER,
        max_new_tokens=MAX_TOKENS
    )
    
    print("\n[STEERED OUTPUT]")
    print(steered_response)
    
    print("\nWriting output to torgenesis_log.txt...")
    with open("torgenesis_log.txt", "w", encoding="utf-8") as f:
        f.write("================ BASELINE (NO STEERING) ================\n")
        f.write(baseline_response + "\n\n")
        f.write("================ STEERED (TOROID) ================\n")
        f.write(steered_response + "\n")
        
    print("Log complete.")