import sys
sys.path.append('/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/')
from metamodel import HRNetMetaModel
import torch
import cv2


def prepare_tensor():
    image = cv2.imread('/scratch-shared/ivanmiert/final_events_folder/frames_10000/event_2135530621/frame_0055.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]
    return tensor

checkpoint_path = '/scratch-shared/ivanmiert/data/model_finals_full.pt' 
model = torch.load(checkpoint_path)

model.eval()

input_tensor = prepare_tensor()
#Perform Inference
with torch.no_grad():
    output = model.predict(input_tensor)
print(output)

file_path = '/scratch-shared/ivanmiert/tensor_output_firsttest.pt'
torch.save(output, file_path)
