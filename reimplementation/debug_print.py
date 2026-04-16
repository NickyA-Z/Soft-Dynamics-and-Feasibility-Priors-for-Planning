import torch, pickle

base = "/home/scur0196/DL2---Grounding-Generated-Videos-/dino_wm/data/pusht_noise/train"

abs_actions = torch.load(f"{base}/abs_actions.pth")
states = torch.load(f"{base}/states.pth")
velocities = torch.load(f"{base}/velocities.pth")

with open(f"{base}/seq_lengths.pkl", "rb") as f:
    seq_lengths = pickle.load(f)

print(type(abs_actions))
print(type(states))
print(type(velocities))
print(type(seq_lengths))

if hasattr(abs_actions, "shape"):
    print("abs_actions shape:", abs_actions.shape)
if hasattr(states, "shape"):
    print("states shape:", states.shape)
print("seq_lengths example:", seq_lengths[:5] if hasattr(seq_lengths, "__getitem__") else seq_lengths)
