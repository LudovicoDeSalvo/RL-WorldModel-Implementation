import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cma
import os
from collections import deque

# --- HELPER FUNCTIONS ---

def get_flat_params(model):
    
    params = []

    for param in model.parameters():
        params.append(param.view(-1))

    return torch.cat(params)

def set_flat_params(model, flat_params):

    prev_ind = 0

    for param in model.parameters():

        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size())
        )
        prev_ind += flat_size

# --- MODEL COMPONENTS ---

class VAE(nn.Module):
    """
    V Model
    input: 3 x 48 x 48
    """
    def __init__(self, latent_dim=32):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # 48x48 to 3x3
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), 
            nn.ReLU()
        )
        
        self.flatten_dim = 256 * 3 * 3 
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # 3x3 to 48x48
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), 
            nn.Sigmoid() 
        )

    def encode(self, x):

        h = self.encoder(x)
        h = h.view(-1, self.flatten_dim)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z):

        h = self.decoder_input(z)
        h = h.view(-1, 256, 3, 3) 

        return self.decoder(h)

    def forward(self, x):

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        return recon, mu, logvar

class Controller(nn.Module):

    def __init__(self, latent_dim=32, action_dim=3, stack_size=3, hidden_dim=64):
        super(Controller, self).__init__()

        input_size = latent_dim * stack_size
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, z_stacked):
        return torch.tanh(self.net(z_stacked))

# --- POLICY ---

class Policy(nn.Module):

    continuous = True 

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.latent_dim = 32
        self.stack_size = 3 
        
        self.vae = VAE(latent_dim=self.latent_dim)
        self.controller = Controller(latent_dim=self.latent_dim, action_dim=3, stack_size=self.stack_size)
        
        self.latent_buffer = deque(maxlen=self.stack_size)
        
        self.to(self.device)
        print(f"WorldModel policy initialized on {self.device} (stack size: {self.stack_size})")

    def reset_buffer(self):
        self.latent_buffer.clear()

    def act(self, state):
        
        state_resized = state[::2, ::2, :] 
        state_tensor = torch.from_numpy(state_resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0
        
        with torch.no_grad():
            mu, _ = self.vae.encode(state_tensor)
            z_current = mu.cpu().numpy()[0]
            
        if len(self.latent_buffer) == 0:
            for _ in range(self.stack_size):
                self.latent_buffer.append(z_current)
        else:
            self.latent_buffer.append(z_current)
            
        z_stacked_np = np.concatenate(self.latent_buffer)
        z_stacked_tensor = torch.from_numpy(z_stacked_np).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.controller(z_stacked_tensor).cpu().numpy()[0]
        
        steering = action[0]
        gas = (action[1] + 1) / 2.0
        brake = (action[2] + 1) / 2.0
        if gas > 0.5 : brake = 0.0
            
        return np.array([steering, gas, brake])

    def train_vae_phase(self, env):

        print("--- VAE TRAINING ---")
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        
        batch_size = 32
        total_frames = 10000
        data_buffer = []
        
        print(f"Collecting {total_frames} frames of random rollout data")

        s, _ = env.reset()

        while len(data_buffer) < total_frames:

            a = env.action_space.sample()
            s, r, terminated, truncated, _ = env.step(a)
            
            s_resized = s[::2, ::2, :] / 255.0
            s_transposed = np.transpose(s_resized, (2, 0, 1))
            data_buffer.append(s_transposed)
            
            if terminated or truncated:
                s, _ = env.reset()
        
        data_tensor = torch.tensor(np.array(data_buffer), dtype=torch.float32).to(self.device)
        print("VAE Training started")
        
        epochs = 15
        self.vae.train()
        
        for epoch in range(epochs):

            perm = torch.randperm(data_tensor.size(0))
            epoch_loss = 0
            
            for i in range(0, data_tensor.size(0), batch_size):

                indices = perm[i:i+batch_size]
                batch = data_tensor[indices]
                
                optimizer.zero_grad()
                recon, mu, logvar = self.vae(batch)
                
                mse_loss = F.mse_loss(recon, batch, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = mse_loss + kl_loss
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            print(f"  VAE epoch {epoch+1}/{epochs} | Avg loss: {epoch_loss / len(data_buffer):.2f}")
        
        print("VAE Training Complete")

        for param in self.vae.parameters():
            param.requires_grad = False

        self.vae.eval()
        self.save()

    def train_controller_phase(self, env):

        print("--- CONTROLLER TRAINING ---")
        
        init_params = get_flat_params(self.controller).detach().cpu().numpy()
        es = cma.CMAEvolutionStrategy(init_params, 0.2, {'popsize': 64})
        
        generations = 150
        global_best_validated_score = -float('inf')
        
        for g in range(generations):

            solutions = es.ask()
            rewards = []
            
            for params in solutions:

                set_flat_params(self.controller, torch.tensor(params).float().to(self.device))
                
                total_reward = 0
                s, _ = env.reset()
                self.reset_buffer()
                
                done, truncated = False, False
                steps = 0
                
                while not (done or truncated) and steps < 800:
                    action = self.act(s)
                    s, r, done, truncated, _ = env.step(action)
                    if r < 0: r *= 1.5 
                    total_reward += r
                    steps += 1
                    if total_reward < -20: break
                
                rewards.append(total_reward)
            
            es.tell(solutions, [-r for r in rewards])
            
            current_generation_best_score = max(rewards)
            best_idx = np.argmax(rewards)
            
            print(f"Gen {g+1}: Raw best: {current_generation_best_score:.2f}, Mean: {np.mean(rewards):.2f}")
            
            if current_generation_best_score > global_best_validated_score:
                print(f"  - Potential new record. Validating on 10 tracks")
                
                candidate_params = solutions[best_idx]
                set_flat_params(self.controller, torch.tensor(candidate_params).float().to(self.device))
                
                val_scores = []

                for _ in range(10): 

                    val_r = 0
                    s, _ = env.reset()
                    self.reset_buffer()
                    done, truncated = False, False
                    steps = 0

                    while not (done or truncated) and steps < 800:

                        action = self.act(s)
                        s, r, done, truncated, _ = env.step(action)
                        val_r += r
                        steps += 1

                    val_scores.append(val_r)
                
                avg_val_score = np.mean(val_scores)
                print(f"  - Validation score: {avg_val_score:.2f}")
                
                if avg_val_score > global_best_validated_score:
                    global_best_validated_score = avg_val_score
                    self.save()
                    print(f"  - NEW BEST! Model saved")
                else:
                    print(f"  - Failed validation")

    def train(self):

        env = gym.make('CarRacing-v2', continuous=True)

        if not os.path.exists('model.pt'):
            self.train_vae_phase(env)
        else:
            print("Model found. Skipping VAE training")
            self.load()
            
        self.train_controller_phase(env)
        print("--- TRAINING COMPLETE ---")

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):

        try:
            self.load_state_dict(torch.load('model.pt', map_location=self.device))
            print("Model loaded")

        except FileNotFoundError:
            print("No saved model found")

    def to(self, device):
        
        ret = super().to(device)
        ret.device = device
        return ret