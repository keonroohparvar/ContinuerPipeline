# ContinuerPipeline
This repo contains most of the work done by Keon Roohparvar in for his Master's Thesis. The Continuer Pipeline is a pipeline that utilizes a novel Latent Diffusion model architecture to take a piece of music and extend it by 5 seconds. 

The pipeline is implemented at a high level in the `continuer_pipeline.py` script, and it extends the [`DiffusionPipeline`](https://huggingface.co/docs/diffusers/v0.17.1/en/api/diffusion_pipeline#diffusers.DiffusionPipeline) class from HuggingFace to allow ease of use. 

The file structure of this repo is the following:
```
.
├── README.md              
├── continuer_pipeline.py  # The main file that contains the pipeline implementation
├── data                   # 
├── legacy
└── results
```
