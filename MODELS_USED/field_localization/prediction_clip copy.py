import sys
sys.path.append('/home/ivanmiert/sportlight_folder/soccernet-calibration-sportlight/')
import cv2
import os
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import concurrent.futures


def transform_image_tensor(image_path):
    # Transforms an image into a tensor ready to use in the model
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]
    return tensor

def batch_images_to_tensors(base_image_folder, batch_size=32):
    # Takes a batch of images (in one folder) and creates batches of tensors
    image_files = sorted(os.listdir(base_image_folder))
    
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        tensor_list = []
        for image_file in batch_files:
            image_path = os.path.join(base_image_folder, image_file)
            tensor = transform_image_tensor(image_path)
            tensor_list.append(tensor)
        yield tensor_list  # Yield a batch of tensors

def get_predictions(tensor_list, model, device):
    predictions = []
    batch_tensor = torch.cat(tensor_list, dim=0).to(device)  # Concatenate and move to device
    with torch.no_grad():
        if hasattr(model, 'nn_module'):
            batch_predictions = model.nn_module(batch_tensor)
        else:
            batch_predictions = model(batch_tensor)
        for prediction in batch_predictions:
            predictions.extend(prediction.cpu().tolist())
        #predictions.extend(batch_predictions.cpu().tolist())  # Store predictions and move to CPU
    return predictions

# def process_clip_folder(clip_folder, image_base_folder, model, output_base_folder, device, batch_size=2):
#     # Model is already on GPU, no need to move it here
#     print('derde summary:')
#     print(torch.cuda.memory_summary(device=device, abbreviated=False))
#     clip_folder_path = os.path.join(image_base_folder, clip_folder)
#     if os.path.isdir(clip_folder_path):
#         prediction_output_folder = os.path.join(output_base_folder, clip_folder)
#         output_file = os.path.join(prediction_output_folder, 'predictions.pth')
        
#         if os.path.exists(output_file):
#             print(f"Skipping {clip_folder} as predictions already exist.")
#             print('summary als wordt geskipt omdat bestaat:')
#             print(torch.cuda.memory_summary(device=device, abbreviated=False))
#             return
        
#         print(f"Predicting for folder: {clip_folder}")
#         predictions = []
#         for tensor_batch in batch_images_to_tensors(clip_folder_path, batch_size):
#             try:
#                 batch_predictions = get_predictions(tensor_batch, model, device)
#                 predictions.extend(batch_predictions)
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print(f"CUDA out of memory error for batch size {batch_size}. Reducing batch size.")
#                     torch.cuda.empty_cache()
#                     smaller_batch_size = batch_size // 2
#                     if smaller_batch_size < 1:
#                         raise RuntimeError("Batch size too small to continue processing.")
#                     for tensor_batch in batch_images_to_tensors(clip_folder_path, smaller_batch_size):
#                         batch_predictions = get_predictions(tensor_batch, model, device)
#                         predictions.extend(batch_predictions)
#                 else:
#                     raise e
#             torch.cuda.empty_cache() #Clear GPU cache after each batch
#         print('summary als hij heeft predict on the folder:')
#         print(torch.cuda.memory_summary(device=device, abbreviated=False))
        
#         os.makedirs(prediction_output_folder, exist_ok=True)
#         torch.save(predictions, output_file)
#         del predictions
#         torch.cuda.empty_cache()
#         print(f"Predictions for {clip_folder} saved to {output_file}")

# def perform_on_subfolders(image_base_folder, model, output_base_folder, batch_size, device):
#     clip_folders = sorted(os.listdir(image_base_folder))
#     print('nog ff 1 tussendoor hoor:')
#     print(torch.cuda.memory_summary(device=device, abbreviated=False))
#     # Using torch.multiprocessing with a specified number of workers
#     processes = []
#     for clip_folder in clip_folders:
#         p = mp.Process(target=process_clip_folder, args=(clip_folder, image_base_folder, model, output_base_folder, device, batch_size))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

# if __name__ == '__main__':
#     mp.set_start_method('spawn')
#     image_base_folder = '/scratch-shared/ivanmiert/final_events_folder/frames'
#     output_base_folder = '/scratch-shared/ivanmiert/final_events_folder/calibration2'
#     model_path = '/scratch-shared/ivanmiert/data/model_finals_full.pt'
    
#     # Load model in the main process and share it across subprocesses
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     print('eerste summary:')
#     print(torch.cuda.memory_summary(device=device, abbreviated=False))
#     model = torch.load(model_path, map_location=device)  # Load model on GPU
#     print('tweede summary:')
#     print(torch.cuda.memory_summary(device=device, abbreviated=False))
#     model.share_memory()  # Share the model across processes
#     batch_size = 2
#     print('Starting processing:')
#     perform_on_subfolders(image_base_folder, model, output_base_folder, batch_size, device)


def process_clip_folder(clip_folder, image_base_folder, model, output_base_folder, device, batch_size=32):
    print(batch_size)
    #print(f"Processing folder: {clip_folder}")
    #print(torch.cuda.memory_summary(device=device, abbreviated=False))
    clip_folder_path = os.path.join(image_base_folder, clip_folder)
    if os.path.isdir(clip_folder_path):
        prediction_output_folder = os.path.join(output_base_folder, clip_folder)
        output_file = os.path.join(prediction_output_folder, 'predictions.pth')
        
        if os.path.exists(output_file):
            print(f"Skipping {clip_folder} as predictions already exist.")
            return
        
        predictions = []
        for tensor_batch in batch_images_to_tensors(clip_folder_path, batch_size):
            try:
                batch_predictions = get_predictions(tensor_batch, model, device)
                predictions.extend(batch_predictions)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA out of memory error for batch size {batch_size}. Reducing batch size.")
                    torch.cuda.empty_cache()
                    smaller_batch_size = batch_size // 2
                    if smaller_batch_size < 1:
                        raise RuntimeError("Batch size too small to continue processing.")
                    for tensor_batch in batch_images_to_tensors(clip_folder_path, smaller_batch_size):
                        batch_predictions = get_predictions(tensor_batch, model, device)
                        predictions.extend(batch_predictions)
                else:
                    raise e
            #print('before cache:')
            #print(torch.cuda.memory_summary(device=device, abbreviated=False))
            torch.cuda.empty_cache() # Clear GPU cache after each batch
            #print('after cache:')
            #print(torch.cuda.memory_summary(device=device, abbreviated=False))
        
        os.makedirs(prediction_output_folder, exist_ok=True)
        torch.save(predictions, output_file)
        del predictions
        torch.cuda.empty_cache()
        print(f"Predictions for {clip_folder} saved to {output_file}")

if __name__ == '__main__':
    image_base_folder = '/scratch-shared/ivanmiert/final_events_folder/frames'
    output_base_folder = '/scratch-shared/ivanmiert/final_events_folder/calibration3'
    model_path = '/scratch-shared/ivanmiert/data/model_finals_full.pt'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model once and move to GPU
    model = torch.load(model_path, map_location=device)
    model.eval()
    print("Model loaded.")
    print(torch.cuda.memory_summary(device=device, abbreviated=False))
    
    clip_folders = sorted(os.listdir(image_base_folder))
    print(clip_folders)
    for clip_folder in clip_folders:
        process_clip_folder(clip_folder, image_base_folder, model, output_base_folder, device, batch_size=4)