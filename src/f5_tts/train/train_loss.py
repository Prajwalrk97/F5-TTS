import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

def load_tensorboard_data(event_file: str):
    """Load loss data from a single TensorBoard event file."""
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    
    if 'loss' not in event_acc.scalars.Keys():
        return [], []
        
    loss_events = event_acc.Scalars('loss')
    steps = [event.step for event in loss_events]
    losses = [event.value for event in loss_events]
    
    valid_data = [(s, l) for s, l in zip(steps, losses) if not np.isnan(l)]
    if valid_data:
        steps, losses = zip(*valid_data)
    
    return list(steps), list(losses)

def plot_training_progress(tensorboard_path: str):
    """Create a loss plot from multiple TensorBoard log files."""
    event_files = glob.glob(os.path.join(tensorboard_path, "events.out.tfevents.*"))
    if not event_files:
        print(f"No TensorBoard event files found in {tensorboard_path}")
        return
    
    all_steps = []
    all_losses = []
    
    for event_file in sorted(event_files):
        steps, losses = load_tensorboard_data(event_file)
        all_steps.extend(steps)
        all_losses.extend(losses)
    
    if not all_losses:
        print("No loss data found in TensorBoard logs")
        return
    
    steps = np.array(all_steps)
    losses = np.array(all_losses)
    valid_mask = ~np.isnan(losses)
    steps = steps[valid_mask]
    losses = losses[valid_mask]
    
    plt.figure(figsize=(20, 10))  # Increased figure size for better readability
    
    # Plot raw loss values
    plt.plot(steps, losses, 'lightblue', alpha=0.3, linewidth=1, label='Raw Loss')
    
    # Plot smoothed loss curve
    if len(losses) > 3:
        window_length = min(len(losses) // 3, 201)
        window_length = window_length if window_length % 2 == 1 else window_length - 1
        smoothed_losses = savgol_filter(losses, window_length, 3)
        plt.plot(steps, smoothed_losses, 'blue', linewidth=2, label='Smoothed Loss')
    
    # Calculate statistics
    min_loss = np.min(losses)
    max_loss = np.max(losses)
    mean_loss = np.mean(losses)
    final_loss = losses[-1]
    
    # Add statistics to plot
    stats_text = (f'Min Loss: {min_loss:.4f}\n'
                 f'Max Loss: {max_loss:.4f}\n'
                 f'Mean Loss: {mean_loss:.4f}\n'
                 f'Final Loss: {final_loss:.4f}')
    
    plt.text(0.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set x-axis ticks every 4000 steps
    start_step = (steps[0] // 4000) * 4000
    end_step = ((steps[-1] // 4000) + 1) * 4000
    tick_positions = np.arange(start_step, end_step + 1, 4000)
    plt.xticks(tick_positions, rotation=45, ha='right')
    
    # Add minor gridlines
    plt.grid(True, which='both', alpha=0.3)
    plt.grid(True, which='major', alpha=0.5)
    
    plt.title('F5-TTS Training Loss Over Time')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.legend()
    
    # Adjust layout to prevent tick labels from being cut off
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Total training steps: {len(steps)}")
    print(f"Loss range: {min_loss:.4f} - {max_loss:.4f}")
    print(f"Mean loss: {mean_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")

if __name__ == "__main__":
    tensorboard_path = r".\runs\F5TTS_Base"
    plot_training_progress(tensorboard_path)