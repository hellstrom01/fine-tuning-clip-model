import matplotlib.pyplot as plt
import json
import os

def load_dataset(dataset_path, classes=False, txt_dir="data/txtclasses_rsicd"):
    """
    Loads the dataset and optionally appends class labels based on the txtclasses folder.
    
    Args:
        dataset_path (str): Path to the dataset_rsicd.json file.
        split (str): Which split to return ('train', 'val', or 'test').
        classes (bool): If True, parses the txt files and appends a 'class' key.
        txt_dir (str): Path to the folder containing the class .txt files.
    """
    # Good practice: use the variable passed into the function
    with open(dataset_path, "r") as f:
        data = json.load(f)
        
    if classes:
        # 1. Build a fast lookup dictionary mapping filename -> class_name
        filename_to_class = {}
        
        # Go through every .txt file in the directory
        for txt_filename in os.listdir(txt_dir):
            if txt_filename.endswith(".txt"):
                # The class name is the file name without the last 4 characters (".txt")
                class_name = txt_filename[:-4] 
                
                # Open the text file and read the image names
                txt_path = os.path.join(txt_dir, txt_filename)
                with open(txt_path, "r") as f:
                    # Read lines, strip whitespace/newlines, and ignore empty lines
                    image_names = [line.strip() for line in f.readlines() if line.strip()]
                    
                    # Map each image name to this class
                    for img_name in image_names:
                        filename_to_class[img_name] = class_name
                        
        # 2. Assign the classes to the JSON data in a single pass
        for item in data["images"]:
            img_name = item["filename"]
            if img_name in filename_to_class:
                item["class"] = filename_to_class[img_name]
            else:
                item["class"] = "Unknown" # Safety fallback
                
    # 3. Filter the dataset by the requested split
    splitA = [i for i in data["images"] if i["split"] == "train"] # (Unlabeled dataset) Not allowed to use sentences
    splitB = [i for i in data["images"] if i["split"] == "val"]   # (Labeled dataset) Allowed to use sentences
    splitC = [i for i in data["images"] if i["split"] == "test"]  # Actual tess dataset for evaluating performance

    return splitA, splitB, splitC
        



def plot_training_results(history, title=None, save=None):
    """
    Plots training components, generalization, and zero-shot accuracy.
    """
    epochs = range(1, len(history['total_losses']) + 1)
    
    plt.style.use('ggplot') 
    # Increased figsize to accommodate the third plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))

    # --- Plot 1: Breakdown of Training Loss Components ---
    ax1.plot(epochs, history['supervised_losses'], 'o-', label='Supervised (Split B)', color='#3498db', linewidth=2)
    ax1.plot(epochs, history['unsupervised_losses'], 's-', label='Unsupervised (Split A)', color='#e67e22', linewidth=2)
    ax1.plot(epochs, history['total_losses'], 'd--', label='Total Training Loss', color='#2ecc71', alpha=0.7)
    
    ax1.set_title('Training Loss Components', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.set_xticks(epochs)
    ax1.legend(frameon=True, facecolor='white')

    # --- Plot 2: Generalization (Loss) ---
    ax2.plot(epochs, history['total_losses'], 'o-', label='Train Loss', color='#2ecc71', linewidth=2)
    ax2.plot(epochs, history['validation_losses'], 'x-', label='Val Loss', color='#e74c3c', linewidth=2)
    
    ax2.set_title('Loss Generalization', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss Value', fontsize=12)
    ax2.set_xticks(epochs)
    ax2.legend(frameon=True, facecolor='white')

    # Annotate Best (Minimum) Val Loss
    min_val_loss = min(history['validation_losses'])
    min_val_epoch = epochs[history['validation_losses'].index(min_val_loss)]
    ax2.annotate(f'Best: {min_val_loss:.3f}', 
                 xy=(min_val_epoch, min_val_loss), 
                 xytext=(min_val_epoch, min_val_loss + 0.3),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 horizontalalignment='center')

    # --- Plot 3: Downstream Performance (Zero-Shot Accuracy) ---
    # Multiply by 100 for percentage view
    accuracies = [acc * 100 for acc in history['epoch_zs_accuracies']]
    ax3.plot(epochs, accuracies, 'o-', color='#9b59b6', linewidth=3, markersize=8)
    
    ax3.set_title('Zero-Shot Accuracy (Split C)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax3.set_xticks(epochs)
    ax3.set_ylim(0, max(accuracies) + 10) # Dynamic scale with some headroom

    # Annotate Best (Maximum) Accuracy
    max_acc = max(accuracies)
    max_acc_epoch = epochs[accuracies.index(max_acc)]
    ax3.annotate(f'Peak: {max_acc:.1f}%', 
                 xy=(max_acc_epoch, max_acc), 
                 xytext=(max_acc_epoch, max_acc - 10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 horizontalalignment='center', color='#8e44ad', fontweight='bold')

    suptitle = title if title else 'CLIP Semi-Supervised Fine-Tuning Metrics' 
    plt.suptitle(suptitle, fontsize=20, y=1.05)
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save}")

    plt.show()