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


# # Local imports
# from data.util.SpectrogramConverter import SpectrogramConverter
# from data.util.SpectrogramParams import SpectrogramParams

class ContinuerPipeline(DiffusionPipeline):
    def __init__(
        self, 
        unet: UNet2DModel, 
        scheduler: Union[DDPMScheduler, DDIMScheduler],
        vqvae: AutoencoderKL,
        mel: Mel,
        generator_name: Optional[str] = None
    ):
        super().__init__()
        self.register_modules(
            vqvae=vqvae,
            unet=unet,
            mel=mel,
            scheduler=scheduler
        )

        self.generator_name = generator_name

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
    
    def _encode_img(self, img: Image.Image, generator: torch.Generator):
        input_image = np.frombuffer(img.tobytes(), dtype="uint8").reshape(
            (img.height, img.width)
        )
        input_image = (input_image / 255) * 2 - 1
        imgs = torch.tensor(input_image[np.newaxis, :, :], dtype=torch.float).to(self.device)
        imgs = self.vqvae.encode(torch.unsqueeze(imgs, 0)).latent_dist.sample(
            generator=generator
        )[0]
        imgs = self.vqvae.config.scaling_factor * imgs
        
        return imgs[np.newaxis, :, :]

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        pipeline = ContinuerPipeline.from_pretrained(
            checkpoint,
            revision='main',
            torch_dtype=dtype
        )

        model = pipeline.to(device)

        return model

    @torch.no_grad()
    def generate_init_image(
        self,
        diffusion_pipeline_name: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        out_path: Optional[str] = None
    ) -> Image.Image :
        if diffusion_pipeline_name is None:
            DEFAULT_AUDIO_DIFFUSION_GENERATOR = 'teticio/audio-diffusion-256'
            pipe = DiffusionPipeline.from_pretrained(DEFAULT_AUDIO_DIFFUSION_GENERATOR).to(self.device)
            print('using default pipe!')
        else:
            pipe = DiffusionPipeline.from_pretrained(diffusion_pipeline_name).to(self.device)
        
        seed = generator.seed()
        print(f'Seed = {seed}')
        generator.manual_seed(seed)

        pipe_output = pipe(generator=generator)
        img_output = pipe_output.images[0]

        # pipe_output = pipe(
        #     audio_file='./data/Breezin.wav',
        #     slice=5,
        #     generator=generator
        # # )
        # ).images[0]

        print(f'Init image generated:')
        img_numpy = np.asarray(img_output)
        print(img_numpy)
        print(np.max(img_numpy))
        print(np.min(img_numpy))

        if out_path is not None:
            img_output.save(out_path)

        return img_output
    
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
        root_img: Optional[Image.Image] = None,
        prev_img: Optional[Image.Image] = None,
        num_imgs_generated: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        step_generator: Optional[torch.Generator] = None,
        num_inference_steps: int = 50,
        eta: float = 0,
        return_dict: bool = True,
        out_dir: Optional[str] = None,
        **kwargs,
    ):
        r"""
        TODO: FIX THE ARGS 
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

        # For backwards compatability
        if type(self.unet.sample_size) == int:
            self.unet.sample_size = (self.unet.sample_size, self.unet.sample_size)

        if generator == None:
            generator = torch.Generator(
                device=self.unet.device
            )
        
        step_generator = step_generator or generator

        # Configure root_img and prev_img
        if root_img is None:
            root_img = self.generate_init_image(out_path='root_img.jpg', generator=generator)
            
        root_latents = self._encode_img(root_img, generator)
        
        if prev_img is None:
            prev_img = root_img
            prev_latents = root_latents 
        else:
            prev_latents = self._encode_img(prev_img, generator)
        
        # Initialize the generator to a printed seed
        seed = generator.seed()
        print(f'Training Seed: {seed}')
        generator.manual_seed(seed)

        # Loop over the number of images we are going to generate
        output_imgs = []
        for _ in range(num_imgs_generated):
            latents = torch.randn(
                (
                    1,
                    1,
                    self.unet.sample_size[0],
                    self.unet.sample_size[1],
                ),
                generator=generator,
                device=self.device,
            )

            # Set scheduler timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps

            # Do backwards diffusion
            for step, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                # Configure input for model
                model_input = torch.cat([root_latents, prev_latents, latents], axis=1)
                if step == 0:
                    print(f'Model input shape: {model_input.shape}')

                model_output = self.unet(model_input, t)["sample"]

                if step == 0:
                    print('output!')
                    print(model_output)

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
                

            prev_latents = latents
            if self.vqvae is not None:
                # 0.18215 was scaling factor used in training to ensure unit variance
                latents = 1 / 0.18215 * latents
                images = self.vqvae.decode(latents)["sample"]
                print(f'Images decoded shape: {images.shape}')
                output_imgs.append(images[0])
            
        # Format images
        out_pil_images = []
        for out_img in output_imgs:
            image = (out_img/ 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(1, 2, 0).numpy()
            image = (image * 255).round().astype("uint8")
            print(image.shape)
            image = Image.fromarray(image[:, :, 0]) if image.shape[2] == 1 else \
                Image.fromarray(image, mode="RGB").convert("L")

            print('FINAL OUTPUTS')
            print(image)
            out_pil_images.append(image)

        audios = list(map(lambda _: self.mel.image_to_audio(_), out_pil_images))

        # Save results if out_dir is specified
        if out_dir is not None:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            
            # Export images
            for i, img in enumerate(out_pil_images):
                img.save(os.path.join(out_dir, f'img{i}.jpg'))
            
            # Export individual wavs
            for i, wav in enumerate(audios):
                self.export_audio(wav, os.path.join(out_dir, f'slice{i}.wav'))
            
            # Export audio in its entirety
            audio_concatenated = np.concatenate(np.array(audios))
            self.export_audio(audio_concatenated, os.path.join(out_dir,'entire_song.wav'))

        if not return_dict:
            return images, (self.mel.get_sample_rate(), audios)

        return BaseOutput(**AudioPipelineOutput(np.array(audios)[:, np.newaxis, :]), **ImagePipelineOutput(images))


if __name__ == '__main__':
    model_path = './keons_continuer'

    unet = UNet2DModel.from_pretrained(model_path, subfolder='unet')
    vqvae = AutoencoderKL.from_pretrained(model_path, subfolder='vqvae')
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')
    mel = Mel.from_pretrained(model_path, subfolder='mel')

    # path1 = Image.open('root_img.jpg')
    # path2 = Image.open('out.jpg')
    # audios = list(map(lambda _: mel.image_to_audio(_), [path1, path2]))

    pipe = ContinuerPipeline(
        unet = unet,
        scheduler = scheduler,
        vqvae = vqvae,
        mel = mel
    )

    pipe(
        # root_img=Image.open('root_img.jpg'),
        num_imgs_generated=5,
        out_dir='keon_results'
    )

    # # output = pipe(batch_size=1, return_dict=True, num_inference_steps=1000)
    # output = pipe(batch_size=1, return_dict=True)
    # print(output)

    # for img in output["images"]:
    #     print(type(img))
    #     img.show(img)
    #     img.save('yeet.jpg')
    
    # for wav_list in output['audios']:
    #     wav = wav_list[0]
    #     print(wav)
    #     print(wav.shape)
    #     print(wav.dtype)
    #     print(np.min(wav))
    #     print(np.max(wav))
    #     # exit()
    #     pipe.export_audio(wav, './test_keon.wav')



