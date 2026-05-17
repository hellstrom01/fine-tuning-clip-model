import os
import torch
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# --- 1. Target Classes (Must match PDF Page 8 exactly) ---
CLASS_NAMES = [
    'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
    'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
    'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass',
    'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt'
]

class CompetitionDataset(Dataset):
    """Simple dataset to load images from a single folder."""
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        # Find all JPEG files
        self.filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
        print(f"✅ Found {len(self.filenames)} leaderboard images.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, fname

@torch.no_grad()
def run_leaderboard_inference(model_path, data_dir, output_file="predictions.txt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load fresh model architecture
    model, _, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    
    # 2. Load your weights
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # Safety: If you forgot to strip '_orig_mod.' during save, this fix handles it
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # 3. Encode Prompts (The 'Zero-Shot' part)
    print("Encoding category prompts...")
    prompts = [f"A satellite image of a {name}." for name in CLASS_NAMES]
    text_tokens = tokenizer(prompts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)

    # 4. Prepare Dataloader
    dataset = CompetitionDataset(data_dir, preprocess_val)
    # Using 4 workers and batch size 128 for speed
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # 5. Inference Loop
    print("Starting classification...")
    results = []
    for images, fnames in tqdm(dataloader):
        images = images.to(device)
        
        # Encode Images
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        # Calculate Cosine Similarity
        # [Batch, 512] @ [512, 21] -> [Batch, 21]
        similarity = image_features @ text_features.T
        
        # Get top prediction index
        preds = similarity.argmax(dim=-1)
        
        # Map indices back to class names and store
        for i in range(len(fnames)):
            predicted_class = CLASS_NAMES[preds[i]]
            results.append(f"{fnames[i]} {predicted_class}")

    # 6. Save results to disk
    with open(output_file, "w") as f:
        # Join lines with newlines; no trailing comma or header
        f.write("\n".join(results))
    
    print(f"\n✨ Done! Generated {len(results)} predictions in {output_file}")

# --- EXECUTION ---
if __name__ == "__main__":
    # Update these paths to your actual locations
    MY_MODEL = "/nobackup/marfr380/models/clip_best_model_2.pt"
    LEADERBOARD_IMAGES = "data/Leaderboard_data"
    
    run_leaderboard_inference(MY_MODEL, LEADERBOARD_IMAGES)