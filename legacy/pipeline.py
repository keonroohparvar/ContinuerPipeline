# Adapted from https://github.com/teticio/audio-diffusion/blob/main/audiodiffusion/pipeline_audio_diffusion.py

from typing import Optional, Tuple, Union
import os

from PIL import Image
import PIL
import numpy as np
import torch
import soundfile as sf

from diffusers import (
    DiffusionPipeline,
    AudioPipelineOutput,
    ImagePipelineOutput
)
from diffusers.utils import BaseOutput
from diffusers.models import AutoencoderKL, UNet2DModel
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from audiodiffusion.mel import Mel


# Local imports
from data.util.SpectrogramConverter import SpectrogramConverter
from data.util.SpectrogramParams import SpectrogramParams

class JazzPipeline(DiffusionPipeline):
    def __init__(
        self, 
        unet: UNet2DModel, 
        scheduler: Union[DDPMScheduler, DDIMScheduler],
        vqvae: AutoencoderKL,
        mel: Mel
    ):
        super().__init__()
        self.register_modules(
            vqvae=vqvae,
            unet=unet,
            mel=mel,
            scheduler=scheduler
        )

    def get_input_dims(self) -> Tuple:
        """Returns dimension of input image
        Returns:
            `Tuple`: (height, width)
        """
        input_module = self.vqvae if self.vqvae is not None else self.unet
        # For backwards compatibility
        sample_size = (
            (input_module.sample_size, input_module.sample_size)
            if type(input_module.sample_size) == int
            else input_module.sample_size
        )
        return sample_size

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        pipeline = JazzPipeline.from_pretrained(
            checkpoint,
            revision='main',
            torch_dtype=dtype
        )

        model = pipeline.to(device)

        return model

    @torch.no_grad()
    def generate_init_image(
        self,
        model_path: Optional[str] = None
    ):
    # TODO(keon): Generate initial image
        if model_path is None:
            if self.model_path is None:
                raise NotImplementedError()
            model_path = self.model_path
        
        sd_pipe = JazzPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        sd_pipe.to('cuda')

        image = sd_pipe().images[0]
        self.init_image_location = 'jazz_init_image.png'
        image.save(self.init_image_location)

        return self.init_image_location
    
    @classmethod
    def convert_image_to_wav(
        cls,
        img: Union[str, PIL.Image.Image],
        out_location: Optional[str] = None
    ):
        """
        Converts a spectrogram image to a waveform
        """
        if type(img) == str:
            if not os.path.exists(img):
                raise FileNotFoundError(f'The file {img} was not found.')
            
            spec = Image.open(img)
            # Convert to black and white
            spec = spec.convert("L")

        elif type(img) == PIL.Image.Image:
            spec = img
        
        else:
            raise TypeError(f'The type {type(img)} for img is not supported.')
        
        spec = np.array([np.asarray(spec)], dtype=np.float32)
        print(spec.shape)
        
        params = SpectrogramParams(sample_rate=22050)
        spectrogram_converter = SpectrogramConverter(params)

        wav_segment = spectrogram_converter.audio_from_spectrogram(spec)

        if out_location is not None:
            wav_segment.export(out_f=out_location, format='wav')

        return wav_segment
    
    @torch.no_grad()
    def export_audio(
        self,
        audio: np.ndarray,
        out_path: str
    ):
        """
        This will take raw audio and export it to the location defined by str
        """
        sf.write(out_path, audio, self.mel.get_sample_rate(), 'PCM_24')

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        step_generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 50,
        eta: float = 0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        # if init_image is None:
        #     init_image = self.generate_init_image()
        
        # elif not os.path.exists(init_image):
        #     raise FileNotFoundError(f'The initial file {init_image} is not found.')

        if generator == None:
            generator = torch.Generator(
                device=self.unet.device
            )
        
        step_generator = step_generator or generator

        # Get initial random latents
        latents = torch.randn(
            (
                batch_size,
                self.unet.in_channels,
                self.unet.sample_size[0],
                self.unet.sample_size[1],
            ),
            generator=generator,
            device=self.device,
        )


        # Set scheduler timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # For backwards compatability
        if type(self.unet.sample_size) == int:
            self.unet.sample_size = (self.unet.sample_size, self.unet.sample_size)


        for step, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            model_output = self.unet(latents, t)["sample"]

            if isinstance(self.scheduler, DDIMScheduler):
                latents = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=latents,
                    eta=eta,
                    generator=step_generator,
                )["prev_sample"]
            else:
                latents = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=latents,
                    generator=step_generator,
                )["prev_sample"]
            
        if self.vqvae is not None:
            # 0.18215 was scaling factor used in training to ensure unit variance
            latents = 1 / 0.18215 * latents
            images = self.vqvae.decode(latents)["sample"]
        
        # print('max:')
        # print(torch.max(images))
        # print('min')
        # print(torch.min(images))
        
        # Format images
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = list(
            map(lambda _: Image.fromarray(_[:, :, 0]), images)
            if images.shape[3] == 1
            else map(lambda _: Image.fromarray(_, mode="RGB").convert("L"), images)
        )

        audios = list(map(lambda _: self.mel.image_to_audio(_), images))

        if not return_dict:
            return images, (self.mel.get_sample_rate(), audios)

        return BaseOutput(**AudioPipelineOutput(np.array(audios)[:, np.newaxis, :]), **ImagePipelineOutput(images))


if __name__ == '__main__':
    model_path = '../keons_model'

    unet = UNet2DModel.from_pretrained(model_path, subfolder='unet')
    vqvae = AutoencoderKL.from_pretrained(model_path, subfolder='vqvae')
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')
    mel = Mel.from_pretrained(model_path, subfolder='mel')

    pipe = JazzPipeline(
        unet = unet,
        scheduler = scheduler,
        vqvae = vqvae,
        mel = mel
    )

    # output = pipe(batch_size=1, return_dict=True, num_inference_steps=1000)
    output = pipe(batch_size=1, return_dict=True)
    print(output)

    # for img in output["images"]:
    #     print(type(img))
    #     img.show(img)
    #     img.save('yeet.jpg')
    
    for wav_list in output['audios']:
        wav = wav_list[0]
        print(wav)
        print(wav.shape)
        print(wav.dtype)
        print(np.min(wav))
        print(np.max(wav))
        # exit()
        pipe.export_audio(wav, './test.wav')



