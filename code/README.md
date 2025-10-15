# AutoMPE: Automated Music Performance Evaluation

AutoMPE is a multimodal music performance evaluation framework based on **CAAM (Cross-Attentive Alignment Module)**, **HED (Hierarchical Expressive Decoder)**, and **KGCE (Knowledge-Guided Contrastive Embedding)**. It evaluates piano or music performances on **rhythm, pitch, and expression** using audio, symbolic scores (MIDI), and visual gestures.

## Features
- Note-level rhythmic and pitch alignment (CAAM)
- Multi-scale expression modeling using audio and visual cues (HED)
- Knowledge-guided embedding for rubric-consistent interpretability (KGCE)
- Compatible with MIDI/audio datasets and student performance datasets

## Installation
```bash
git clone https://github.com/yourusername/AutoMPE.git
cd AutoMPE
pip install -r requirements.txt
