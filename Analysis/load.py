import torch
import os

def load_embeddings(map_location='cpu', path=os.path.join("Analysis","Embeddings")):
    h_ts = torch.load(os.path.join(path, "h_ts.pt"), map_location)
    h_fs = torch.load(os.path.join(path, "h_fs.pt"), map_location)
    h_t_augs = torch.load(os.path.join(path, "h_t_augs.pt"), map_location)
    h_f_augs = torch.load(os.path.join(path, "h_f_augs.pt"), map_location)
    z_ts = torch.load(os.path.join(path, "z_ts.pt"), map_location)
    z_fs = torch.load(os.path.join(path, "z_fs.pt"), map_location)
    z_t_augs = torch.load(os.path.join(path, "z_t_augs.pt"), map_location)
    z_f_augs = torch.load(os.path.join(path, "z_f_augs.pt"), map_location)
    return h_ts, z_ts, h_fs, z_fs, h_t_augs, z_t_augs, h_f_augs, z_f_augs

### Example: ###

# from Analysis.load import load_embeddings
# h_ts, z_ts, h_fs, z_fs, h_t_augs, z_t_augs, h_f_augs, z_f_augs = load_embeddings()

h_ts, z_ts, h_fs, z_fs, h_t_augs, z_t_augs, h_f_augs, z_f_augs = load_embeddings()