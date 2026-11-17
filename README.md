# ContinuerPipeline
This repo contains most of the work done for my Master's Thesis. The Continuer Pipeline is a pipeline that utilizes a novel Latent Diffusion model architecture to take a piece of music and extend it by 5 seconds. 

## Inference environment

Use Python 3.12 or newer in a fresh virtual environment:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_for_env.txt
python -m pip check
```

The requirements cover `continuer_pipeline.py`. The scripts under `legacy/`
are archived experiments, not supported by this inference environment. The old
training dependencies (`datasets`, `transformers`, `accelerate`, and
`audiodiffusion`) are no longer installed. The pipeline now uses the Mel
converter included in Diffusers, so it does not need `aiohttp`.
`certifi` remains a required dependency of the HTTP clients used by Diffusers
and Hugging Face; its minimum version excludes the compromised CA bundle.

Create a fresh environment rather than installing over the old frozen environment;
otherwise unused vulnerable packages can remain installed. To audit the resolved
environment, install and run `pip-audit` in that environment.

## Thesis Document
My thesis describes how this technology works in depth, but at a high level, the Continuer Pipeline simply takes in a waveform and predicts what the next 5-second chunk will sound like. It does this using a novel Latent Diffusion model architecture, and internally converts all waveforms to spectrograms to handle this problem in the image space. 

To read the thesis and learn more about the project, please visit the Thesis' official page at the Cal Poly Digital Commons - [A Novel Approach to Extending Music Using Latent Diffusion](https://digitalcommons.calpoly.edu/theses/2670/). 

## Repository Organization


The file structure of this repo is the following:
```
.
├── legacy                 # Contains most of the development/attempted methods to get this project working
├── .gitignore             # Basic Python .gitignore with custom ignores for local data folders
├── results                # Folder with some simple examples of           
├── README.md              # This file
└── continuer_pipeline.py  # The main file that contains the pipeline implementation
```

The pipeline is implemented at a high level in the `continuer_pipeline.py` script, and it extends the [`DiffusionPipeline`](https://huggingface.co/docs/diffusers/v0.17.1/en/api/diffusion_pipeline#diffusers.DiffusionPipeline) class from HuggingFace to allow ease of use. 

My trained models responsible for the diffusion process can be found on the Hugging Face hub [here](https://huggingface.co/keonroohparvar/continuer_pipeline); please refer to the bottom of the `continuer_pipeline.py` file to see an example usage of how to use the pipeline.
