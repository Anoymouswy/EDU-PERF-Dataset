Quick Start
Generate demo dataset
python data/generate_demo_data.py

Train the model
python scripts/train.py --data_path data/demo_data

Evaluate performance
python scripts/evaluate.py --model_path checkpoints/best_model.pth --data_path data/demo_data

License

MIT License


---

## **2. requirements.txt**
```text
torch>=2.1
torchvision
torchaudio
numpy
scipy
matplotlib
pandas
scikit-learn
tqdm
einops
openpose-python
mmpose
umap-learn