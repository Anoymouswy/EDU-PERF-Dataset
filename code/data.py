import os
import numpy as np
import torch

os.makedirs("data/demo_data", exist_ok=True)

num_samples = 10
num_frames = 100
num_notes = 50
audio_dim = 128
visual_dim = 17
score_dim = 3  # pitch, duration, onset

for i in range(num_samples):
    audio = np.random.rand(num_frames, audio_dim).astype(np.float32)
    visual = np.random.rand(num_frames, visual_dim).astype(np.float32)
    score = np.random.rand(num_notes, score_dim).astype(np.float32)
    label_alignment = np.random.randint(0, 5, size=(num_notes,))
    label_expression = np.random.rand(1).astype(np.float32)
    
    np.savez(f"data/demo_data/sample_{i}.npz",
             audio=audio, visual=visual, score=score,
             label_alignment=label_alignment,
             label_expression=label_expression)
print("Demo dataset generated in data/demo_data")
