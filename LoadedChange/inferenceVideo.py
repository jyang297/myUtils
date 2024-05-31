import os
import sys
# import numpy as np
sys.path.append('/root/SE/se/RIFE_LSTM_Context')
root = '/root/SEatt_GRU-RIFE'
output_root = "/root/autodl-tmp"
import torch
import model.toloadRIFE as toload
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# frame_path = output_root + "/outputs/dacing_slowmotion" # typo here I don't want to change it now
frame_path = output_root + "/outputs/soccer_4x" 
# frame_path = "outputs/outputs_slowmotion"                

output_path = output_root + "/outputs/soccer_8x"
pretrained_model_path = root + '/intrain_log'
pretrained_path = root + '/RIFE_log' # pretrained RIFE path
shift = 0



def load_frames(frame_folder, start_frame, num_frames=4):
    # Load a sequence of 'num_frames' starting from 'start_frame'
    frames = []
    for i in range(1,num_frames+1):
        frame_path = os.path.join(frame_folder, f"frame_{start_frame + i:04d}.png")
        frame = Image.open(frame_path).convert('RGB')
        frames.append(frame)
        if i != num_frames:
            frames.append(frame)
        
    # print('load')
    return frames




def preprocess_frames(frames):
    # Convert frames to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # tensor = torch.stack([transform(frame) for frame in frames], dim=1)  # shape: (3, 7, H, W)
    tensor = torch.stack([transform(frame) for frame in frames], dim=0) # shape: (3, 7, H, W)
    #print("Tensor.shape", tensor.shape)
    tensor = tensor.view(1, 3*7, tensor.shape[2], tensor.shape[3])  # shape: (1, 21, H, W)
    
    #print('preprocess')
    return tensor.to(device)

def save_frame(tensor, output_folder, frame_index):
    transform = transforms.ToPILImage()
    img = transform(tensor.cpu())
    img.save(os.path.join(output_folder, f"frame_{frame_index:04d}.png"))
    ##print('save\n')

def inference_video(model, frame_folder, output_folder, total_frames):
    with torch.no_grad():
        for start_frame in range(0,total_frames - 2 + 1, 3):  # Adjust the step to handle overlap or gaps

            # manual shift
            start_frame += shift
            i = int(start_frame/3)
            frames = load_frames(frame_folder, start_frame)
            save_start_point = i*6
            input_tensor = preprocess_frames(frames)
            #print('gointo model')
            output_allframes_tensors = model(input_tensor)
            #print('compute finished')
            interpolated_frames = output_allframes_tensors[:-1] 
            # Example reshape to (1, 7, 3, H, W)
            print('try save')
            # Saving only interpolated frames: indices 0 to 5. the 6th is the 0th of the next saving loop
            for i in range(6):
                save_frame(interpolated_frames[i, :, :, :], output_folder, save_start_point + i + 1)
            torch.cuda.empty_cache()



from model.RIFE import Model

# Load pretrained Optical Flow Model
checkpoint =toload.convertload(torch.load(f'{pretrained_path}/flownet.pkl',map_location=device))
Ori_IFNet_loaded = toload.IFNet_update()
Ori_IFNet_loaded.load_state_dict(checkpoint)
Ori_IFNet_loaded.eval()
for param in Ori_IFNet_loaded.parameters():
    param.requires_grad = False
print("Loaded Pretrained RIFE")

model = Model(Ori_IFNet_loaded, local_rank=0)

model.load_model(pretrained_model_path )
print("Loaded ConvLSTM model")
model.eval()
model.to_device()
inference_video(model.simple_inference, frame_path, output_path, 1200)

