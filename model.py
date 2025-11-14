import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, latent_dim, num_attrs, base_channels=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_attrs = num_attrs
        c = base_channels # 128
        
        # --- Encoder ---
        # Input: (3 + num_attrs) x 64 x 64
        self.enc = nn.Sequential(
            nn.Conv2d(3 + num_attrs, c,   4, 2, 1), nn.LeakyReLU(0.2), # 64 -> 32
            nn.Conv2d(c,   c*2, 4, 2, 1), nn.LeakyReLU(0.2), # 32 -> 16
            nn.Conv2d(c*2, c*4, 4, 2, 1), nn.LeakyReLU(0.2), # 16 -> 8
            nn.Conv2d(c*4, c*8, 4, 2, 1), nn.LeakyReLU(0.2), # 8 -> 4
        )
        # Final encoder feature map size: (c*8) x 4 x 4 = 1024 x 4 x 4
        enc_output_dim = c * 8 * 4 * 4 # 1024 * 16 = 16384
        
        self.fc_mu = nn.Linear(enc_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_output_dim, latent_dim)

        # --- Decoder ---
        self.fc_dec = nn.Linear(latent_dim + num_attrs, enc_output_dim)
        
        self.dec = nn.Sequential(
            # Input: (c*8) x 4 x 4
            nn.ConvTranspose2d(c*8, c*4, 4, 2, 1), nn.ReLU(), # 4 -> 8
            nn.ConvTranspose2d(c*4, c*2, 4, 2, 1), nn.ReLU(), # 8 -> 16
            nn.ConvTranspose2d(c*2, c,   4, 2, 1), nn.ReLU(), # 16 -> 32
            nn.ConvTranspose2d(c,   3,   4, 2, 1),            # 32 -> 64
            nn.Tanh() # Output range [-1, 1]
        )

    def encode(self, x, c):
        # Tile 'c' to match spatial dimensions of 'x'
        c = c.view(-1, self.num_attrs, 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        # Concatenate image and condition
        x_cond = torch.cat([x, c], dim=1)
        
        h = self.enc(x_cond)
        h = h.view(x.size(0), -1) # Flatten
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        # Concatenate latent vector and condition
        z_cond = torch.cat([z, c], dim=1)
        h = self.fc_dec(z_cond)
        h = h.view(-1, 1024, 4, 4) # Reshape to feature map
        return self.dec(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar