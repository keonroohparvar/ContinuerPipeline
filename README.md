# ContinuerPipeline
This repo contains most of the work done for my Master's Thesis. The Continuer Pipeline is a pipeline that utilizes a novel Latent Diffusion model architecture to take a piece of music and extend it by 5 seconds. 

The pipeline is implemented at a high level in the `continuer_pipeline.py` script, and it extends the [`DiffusionPipeline`](https://huggingface.co/docs/diffusers/v0.17.1/en/api/diffusion_pipeline#diffusers.DiffusionPipeline) class from HuggingFace to allow ease of use. 

The file structure of this repo is the following:
```
.
├── legacy                 # Contains most of the development/attempted methods to get this project working
├── .gitignore             # Basic Python .gitignore with custom ignores for local data folders
├── results                # Folder with some simple examples of           
├── README.md              # This file
└── continuer_pipeline.py  # The main file that contains the pipeline implementation
```

My Thesis document describes how this tehcnology works in depth, but at a high level, the Continuer Pipeline simply takes in a waveform and predicts what the next 5-second chunk will sound like. It does this using a novel Latent Diffusion model architecture, and ultimately converts all the waveforms to spectrograms to handle this problem in the image space. 
