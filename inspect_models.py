import torch 

state_dict = torch.load("models/model_tennis_court_det.pt", map_location='cpu')
print(state_dict.keys())