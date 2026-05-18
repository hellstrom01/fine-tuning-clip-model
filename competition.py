import os
import torch
import open_clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# --- 1. Target Classes (Updated to 21 Classes) ---
CLASS_NAMES = [
    'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
    'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
    'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass',
    'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt'
]

# Rich descriptions for the 21 classes to help CLIP tokenization
RICH_DESCRIPTIONS = {
    'agricultural': ['agricultural land', 'farmland', 'cultivated crop fields'],
    'airplane': ['airplane', 'aircraft on the ground'],
    'baseballdiamond': ['baseball diamond', 'baseball field'],
    'beach': ['beach', 'sandy coastline'],
    'buildings': ['buildings', 'urban structures'],
    'chaparral': ['chaparral shrubland', 'desert scrub'],
    'denseresidential': ['dense residential neighborhood', 'closely packed houses'],
    'forest': ['forest', 'dense woodland', 'canopy of trees'],
    'freeway': ['freeway', 'highway', 'multi-lane road'],
    'golfcourse': ['golf course', 'green fairway'],
    'harbor': ['harbor', 'marina with boats', 'port'],
    'intersection': ['road intersection', 'crossroads'],
    'mediumresidential': ['medium residential area', 'suburban neighborhood'],
    'mobilehomepark': ['mobile home park', 'trailer park'],
    'overpass': ['highway overpass', 'bridge over road'],
    'parkinglot': ['parking lot', 'car park'],
    'river': ['river', 'waterway'],
    'runway': ['airport runway', 'landing strip'],
    'sparseresidential': ['sparse residential area', 'rural houses with space'],
    'storagetanks': ['industrial storage tanks', 'oil silos'],
    'tenniscourt': ['tennis court']
}

TEMPLATES = [
    "a satellite image of a {}",
    "an aerial photograph of a {}",
    "a top-down view of a {}",
    "a remote sensing image showing a {}",
    "a centered satellite photo of a {}"
]

class CompetitionDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
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
    
    # 1. Load model architecture
    model, _, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    
    # 2. Load weights
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # 3. Encode Prompts (Rich Descriptions + Templates)
    print("Encoding category prompts with label refinement and ensembling...")
    zeroshot_weights = []
    
    for class_name in CLASS_NAMES:
        descriptions = RICH_DESCRIPTIONS.get(class_name, [class_name])
        texts = []
        for template in TEMPLATES:
            for desc in descriptions:
                texts.append(template.format(desc))
                
        text_tokens = tokenizer(texts).to(device)
        class_embeddings = model.encode_text(text_tokens)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        # Average and re-normalize for a robust class vector
        class_embedding = F.normalize(class_embeddings.mean(dim=0), dim=-1)
        zeroshot_weights.append(class_embedding)

    text_features = torch.stack(zeroshot_weights).to(device)

    # 4. Prepare Dataloader
    dataset = CompetitionDataset(data_dir, preprocess_val)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # 5. Inference Loop
    print("Starting classification...")
    results = []
    for images, fnames in tqdm(dataloader):
        images = images.to(device)
        
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        # Calculate similarity [Batch, 21]
        similarity = image_features @ text_features.T
        preds = similarity.argmax(dim=-1)
        
        for i in range(len(fnames)):
            predicted_class = CLASS_NAMES[preds[i]]
            results.append(f"{fnames[i]} {predicted_class}")

    # 6. Save results
    with open(output_file, "w") as f:
        f.write("\n".join(results))
    
    print(f"\n✨ Done! Generated {len(results)} predictions in {output_file}")

if __name__ == "__main__":
    MY_MODEL = "/nobackup/marfr380/models/clip_best_model_2.pt"
    LEADERBOARD_IMAGES = "data/Leaderboard_data"
    
    run_leaderboard_inference(MY_MODEL, LEADERBOARD_IMAGES)