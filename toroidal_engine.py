import torch
import math

class ToroidalSteeringEngine:
    def __init__(self, latent_dim=2048, device='cuda'):
        """
        Initializes the geometric engine.
        """
        self.latent_dim = latent_dim
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Fixed seed ensures the "geometry" of the latent space remains stable across tests
        torch.manual_seed(42)
        self.projection_matrix = torch.randn(3, latent_dim, device=self.device)

    def get_rotation_matrix_3d(self, pitch, roll, yaw):
        """Calculates the 3D rotation matrix from angles (in radians)."""
        tensor_type = torch.float32
        
        R_x = torch.tensor([[1, 0, 0],
                            [0, math.cos(pitch), -math.sin(pitch)],
                            [0, math.sin(pitch), math.cos(pitch)]], dtype=tensor_type, device=self.device)
                            
        R_y = torch.tensor([[math.cos(roll), 0, math.sin(roll)],
                            [0, 1, 0],
                            [-math.sin(roll), 0, math.cos(roll)]], dtype=tensor_type, device=self.device)
                            
        R_z = torch.tensor([[math.cos(yaw), -math.sin(yaw), 0],
                            [math.sin(yaw), math.cos(yaw), 0],
                            [0, 0, 1]], dtype=tensor_type, device=self.device)
                            
        return torch.mm(R_z, torch.mm(R_y, R_x))

    def generate_steering_vector(self, params, step=0):
        """
        Generates the vector. The 'step' parameter allows the Toroid to rotate dynamically 
        over time based on the defined velocity.
        """
        R = params['major_radius']
        r = params['minor_radius']
        scale = params['scale']
        
        # Calculate dynamic rotation based on the current generation step
        base_x, base_y, base_z = params['rotation_angles']
        vel_x, vel_y, vel_z = params['rotation_velocity']
        
        rot_x = base_x + (vel_x * step)
        rot_y = base_y + (vel_y * step)
        rot_z = base_z + (vel_z * step)

        lines_u = params['line_count_u']
        lines_v = params['line_count_v']
        weights = params['weighting_table']
        center_weight = params['center_weight']

        # 1. Generate Intersections
        theta = torch.linspace(0, 2 * math.pi, lines_u, device=self.device)
        phi = torch.linspace(0, 2 * math.pi, lines_v, device=self.device)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')
        
        theta_flat = theta_grid.flatten()
        phi_flat = phi_grid.flatten()
        
        x = (R + r * torch.cos(phi_flat)) * torch.cos(theta_flat)
        y = (R + r * torch.cos(phi_flat)) * torch.sin(theta_flat)
        z = r * torch.sin(phi_flat)
        
        points_3d = torch.stack((x, y, z), dim=1) 

        # 2. Apply Scaling
        points_3d = points_3d * scale

        # 3. Apply Dynamic Rotation
        rot_matrix = self.get_rotation_matrix_3d(rot_x, rot_y, rot_z)
        points_3d = torch.mm(points_3d, rot_matrix)

        # 4. Project into LLM Latent Space
        latent_intersections = torch.mm(points_3d, self.projection_matrix)

        # 5. Apply the Weighting Table
        total_points = latent_intersections.shape[0]
        weight_tensor = torch.tensor(weights, device=self.device)
        weight_tensor = weight_tensor.repeat(math.ceil(total_points / len(weights)))[:total_points]
        
        weighted_intersections = latent_intersections * weight_tensor.unsqueeze(1)
        summed_intersections = torch.sum(weighted_intersections, dim=0)

        # 6. Process the Center Point
        center_3d = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)
        latent_center = torch.mm(center_3d, self.projection_matrix) * center_weight

        # 7. Final Synthesis
        final_vector = summed_intersections + latent_center.squeeze(0)
        
        return final_vector