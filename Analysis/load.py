import torch
import os

def load_embeddings(map_location='cpu', path=os.path.join("Analysis","Embeddings")):
    h_t = torch.load(os.path.join(path, "h_t.pt"), map_location).detach().numpy()
    h_f = torch.load(os.path.join(path, "h_f.pt"), map_location).detach().numpy()
    h_t_aug = torch.load(os.path.join(path, "h_t_aug.pt"), map_location).detach().numpy()
    h_f_aug = torch.load(os.path.join(path, "h_f_aug.pt"), map_location).detach().numpy()
    z_t = torch.load(os.path.join(path, "z_t.pt"), map_location).detach().numpy()
    z_f = torch.load(os.path.join(path, "z_f.pt"), map_location).detach().numpy()
    z_t_aug = torch.load(os.path.join(path, "z_t_aug.pt"), map_location).detach().numpy()
    z_f_aug = torch.load(os.path.join(path, "z_f_aug.pt"), map_location).detach().numpy()
    return h_t, z_t, h_f, z_f, h_t_aug, z_t_aug, h_f_aug, z_f_aug

### Example: ###

# from load import load_embeddings
# h_t, z_t, h_f, z_f, h_t_aug, z_t_aug, h_f_aug, z_f_aug = load_embeddings()