# Image Retrieval Project

## Overview
This project builds an image retrieval system using a fine-tuned ResNet-18 encoder and FAISS-based similarity search. A web app allows users to upload an image and retrieve visually similar images from the database.

## Project Highlights
- Fine-tuned ResNet-18 for image retrieval
- Applied data augmentation during training
- Achieved 93% validation accuracy
- Built FAISS index for efficient retrieval
- Developed a web app demo for image-based search

## Repository Contents
- `weights/model.pth`: final checkpoint
- `data/faiss_index.bin`: FAISS index
- `data/features_paths.json`: feature-path mapping
- `report/experiment_report.md`: experiment summary
- `demo/`: web app demo examples and outputs
- `webapp/`: application code

## Setup
```bash
mamba env create -f environment.yaml
conda activate w7-image-retrieval
