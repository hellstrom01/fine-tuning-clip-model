import matplotlib.pyplot as plt

def plot_training_results(history, title=None, save=None):
    """
    Plots the training and validation metrics for the semi-supervised CLIP project.
    
    Args:
        history (dict): Dictionary containing the loss lists.
        title (str, optional): Custom title for the plot.
        save (str, optional): File path/name to save the plot (e.g., 'results.png').
    """
    epochs = range(1, len(history['total_losses']) + 1)
    
    # Set a nice theme
    plt.style.use('ggplot') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Breakdown of Training Loss Components ---
    ax1.plot(epochs, history['supervised_losses'], 'o-', label='Supervised (Split B)', color='#3498db', linewidth=2)
    ax1.plot(epochs, history['unsupervised_losses'], 's-', label='Unsupervised (Split A)', color='#e67e22', linewidth=2)
    ax1.plot(epochs, history['total_losses'], 'd--', label='Total Training Loss', color='#2ecc71', alpha=0.7)
    
    ax1.set_title('Training Loss Components', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.set_xticks(epochs)
    ax1.legend(frameon=True, facecolor='white')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Generalization (Total Train vs Val) ---
    ax2.plot(epochs, history['total_losses'], 'o-', label='Total Training Loss', color='#2ecc71', linewidth=2)
    ax2.plot(epochs, history['validation_losses'], 'x-', label='Validation Loss (Split B)', color='#e74c3c', linewidth=2)
    
    ax2.set_title('Model Generalization (Train vs. Val)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss Value', fontsize=12)
    ax2.set_xticks(epochs)
    ax2.legend(frameon=True, facecolor='white')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Annotate the minimum validation loss
    min_val_loss = min(history['validation_losses'])
    min_val_epoch = epochs[history['validation_losses'].index(min_val_loss)]
    ax2.annotate(f'Best: {min_val_loss:.3f}', 
                 xy=(min_val_epoch, min_val_loss), 
                 xytext=(min_val_epoch, min_val_loss + 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 horizontalalignment='center')

    suptitle = title if title else 'Semi-Supervised CLIP Fine-Tuning Performance' 
    plt.suptitle(suptitle, fontsize=18, y=1.02)
    plt.tight_layout()

    # Save logic
    if save:
        # bbox_inches='tight' ensures titles aren't cut off in the saved file
        plt.savefig(save, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save}")

    plt.show()