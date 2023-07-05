# Adapted from https://github.com/teticio/audio-diffusion/blob/main/audiodiffusion/pipeline_audio_diffusion.py

from typing import Optional, Tuple, Union
import os

from PIL import Image
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import soundfile as sf

from diffusers import (
    DiffusionPipeline,
    AudioPipelineOutput,
    ImagePipelineOutput
)
from diffusers.utils import BaseOutput
from diffusers.models import AutoencoderKL, UNet2DModel
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from audiodiffusion.mel import Mel


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
    
    def _encode_img(self, img: Image.Image, generator: torch.Generator):
        """Encodes an image using the VQVAE and applies the correct augmentations.
        Returns:
            `Torch.Tensor`: The original image's latents after being encoded by VQVAE
        
        """
        augmentations = Compose([
            ToTensor(),
            Normalize([0.5], [0.5]),
        ])

        imgs = augmentations(img)

        if len(imgs.shape) == 3:
            imgs = imgs.reshape(-1, *imgs.shape)
        
        imgs = self.vqvae.encode(imgs).latent_dist.sample()
        imgs = imgs * 0.18215

        return imgs

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        """Returns a pretrained `ContinuerPipeline` pipeline to do future training on.

        Returns:
            `ContinuerPipeline`: The pretrained pipeline to do future training on
        
        """

        pipeline = ContinuerPipeline.from_pretrained(
            checkpoint,
            revision='main',
            torch_dtype=dtype
        )

        pipe = pipeline.to(device)

        return pipe

    @torch.no_grad()
    def export_audio(
        self,
        audio: np.ndarray,
        out_path: str
    ):
        """Takes raw audio and exports it to a specified path.
        """
        sf.write(out_path, audio, self.mel.get_sample_rate(), 'PCM_24')

    def get_default_steps(self) -> int:
        """Returns default number of steps recommended for inference

        Returns:
            `int`: number of steps
        """
        return 50 if isinstance(self.scheduler, DDIMScheduler) else 1000

    @torch.no_grad()
    def __call__(
        self,
        audio_path: Optional[str] = None,
        raw_audio: Optional[np.ndarray] = None,
        num_imgs_generated: Optional[int] = 1,
        generator: Optional[torch.Generator] = None,
        step_generator: Optional[torch.Generator] = None,
        steps: Optional[int] = None,
        eta: float = 0,
        return_dict: bool = True,
        out_dir: Optional[str] = None,
        **kwargs,
    ):
        r"""
        Args:
            audio_path (`str`, *optional*): 
                The local path to the .wav file to extend
            raw_audio (`np.ndarray`, *optional*): 
                The raw audio to be extended as a Numpy array

            num_imgs_generated(`int`, *optional*, defaults to 1): 
                How many 5-second increments we want to extend the audio by.
            
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            step_generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make the generation
                deterministic during the steps.

            steps(`int`, *optional*, defaults to 1000): 
                How many denoising steps the Continuer Pipeline will preform. Default (and recommended) is 1000.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).

            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
            out_dir (`str`): 
                The path to where we will save the output of our diffusion process.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        ###############################################################################################################################
        # # # # # INPUT HANDLING # # # # #
        ###############################################################################################################################
        if out_dir is None:
            print(f'Error. Did not provide argument for out_dir.')
            exit()

        if not os.path.exists(os.path.dirname(out_dir)):
            print(f'Error. The directory {os.path.dirname(out_dir)} could not be found.')
            exit()

        steps = steps or self.get_default_steps()
        self.scheduler.set_timesteps(steps)

        # For backwards compatability
        if type(self.unet.sample_size) == int:
            self.unet.sample_size = (self.unet.sample_size, self.unet.sample_size)

        # Initialize generators
        if generator == None:
            generator = torch.Generator(
                device=self.unet.device
            )

        # Initialize the generator to a printed seed
        gen_seed = generator.seed()
        generator.manual_seed(gen_seed)

        # Initialize step generator if not passed in
        step_generator = step_generator or generator

        first_img = None

        # Initialize the generator to a printed seed
        step_seed = step_generator.seed()
        step_generator.manual_seed(step_seed)

        # Print generator seeds
        print(f'\t---> Generator Seed: {gen_seed}')
        if step_generator != generator:
            print(f'\t---> Step Generator Seed: {step_generator}')



        # Get root image from either raw audio or input image
        if audio_path is not None or raw_audio is not None:
            self.mel.load_audio(audio_path, raw_audio)
            number_of_slices = self.mel.get_number_of_slices()
            first_img = self.mel.audio_slice_to_image(slice=0)
            prev_img = self.mel.audio_slice_to_image(slice=(number_of_slices-1))
        
        else:
            print(f'Error. Did not provide audio_path or raw_audio arguments.')
            exit()

        ###############################################################################################################################
        # # # # # DIFFUSION PROCESS # # # # #
        ###############################################################################################################################
        
        # Get x0 and xt latents, and noise latents
        root_latents = self._encode_img(first_img, generator)
        prev_latents = self._encode_img(prev_img, generator)
        target_latents = self._encode_img(
                randn_tensor(
                    first_img.size, 
                    generator=generator,
                    device=self.device
                    ).numpy(),
                generator=generator
        )


        # For debugging purposes
        if os.getenv('DEBUG_CONTINUER', 'false').lower() in ('true', '1', 't'):
            print(f'Min | Max of Root: {torch.min(root_latents)} {torch.max(root_latents)}')
            print(f'Min | Max of prev: {torch.min(prev_latents)} {torch.max(prev_latents)}')
            print(f'Min | Max of prev: {torch.min(target_latents)} {torch.max(target_latents)}')
            print(f'Averages of root, prev, latents: {torch.mean(root_latents)} | {torch.mean(prev_latents)} | {torch.mean(target_latents)}')
            print(f'shapes of root, prev, latents: {root_latents.shape} | {prev_latents.shape} | {target_latents.shape}')


        # Create list for output images
        output_imgs = []

        # Loop over the number of images we are going to generate
        for _ in range(num_imgs_generated):
            target_latents = randn_tensor(
                (
                    1,
                    1,
                    self.unet.sample_size[0],
                    self.unet.sample_size[1],
                ),
                generator=generator,
                device=self.device,
            )

            # Do backwards diffusion
            for step, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                # Configure input for model
                model_input = torch.cat([root_latents, prev_latents, target_latents], axis=1)

                # Debugging print statements to monitor model's performance during diffusion steps
                if os.getenv('DEBUG_CONTINUER', 'false').lower() in ('true', '1', 't'):
                    if step % 100 == 0:
                        print(f'Min|Max|Mean of target latents: {torch.min(target_latents)} | {torch.max(target_latents)} | {torch.mean(target_latents)}')

                # Get model output of predicted noise at timestep t
                model_output = self.unet(model_input, t)["sample"]

                # Perform backwards step with predicted noise based on if DDIM or DDPM scheduler
                if isinstance(self.scheduler, DDIMScheduler):
                    target_latents = self.scheduler.step(
                        model_output=model_output,
                        timestep=t,
                        sample=target_latents,
                        eta=eta,
                        generator=step_generator,
                    )["prev_sample"]
                else:
                    target_latents = self.scheduler.step(
                        model_output=model_output,
                        timestep=t,
                        sample=target_latents,
                        generator=step_generator,
                    )["prev_sample"]
                

            # Get latents representing the final guess and decode them using VQVAE
            prev_latents = target_latents.detach().clone()
            if self.vqvae is not None:
                # 0.18215 was scaling factor used in training to ensure unit variance
                target_latents = 1 / 0.18215 * target_latents
                images = self.vqvae.decode(target_latents)["sample"]
                output_imgs.append(images[0])
            

        
        ###############################################################################################################################
        # # # # # EXPORTING RESULTS # # # # #
        ###############################################################################################################################
        # Format images
        out_pil_images = []
        for out_img in output_imgs:
            image = (out_img/ 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(1, 2, 0).numpy()
            image = (image * 255).round().astype("uint8")
            image = Image.fromarray(image[:, :, 0]) if image.shape[2] == 1 else \
                Image.fromarray(image, mode="RGB").convert("L")

            out_pil_images.append(image)

        audios = list(map(lambda _: self.mel.image_to_audio(_), out_pil_images))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        
        # Export images of synthetic music extensions
        for i, img in enumerate(out_pil_images):
            img.save(os.path.join(out_dir, f'synthetic_img{i}.jpg'))
        
        # Export individual, synthetic waveforms
        for i, wav in enumerate(audios):
            self.export_audio(wav, os.path.join(out_dir, f'synthetic_audio_extension{i}.wav'))
        
        # Export audio in its entirety with original + synthetic portions
        entire_audio_arr = [np.copy(audio) for audio in audios]
        entire_audio_arr.insert(0, self.mel.audio)
        entire_audio_concatenated = np.concatenate(entire_audio_arr)
        self.export_audio(entire_audio_concatenated, os.path.join(out_dir,'extended_song.wav'))

        # For debugging purposes
        if os.getenv('DEBUG_CONTINUER', 'false').lower() in ('true', '1', 't'):
            true_imgs_path = os.path.join(out_dir, 'true_imgs')
            if not os.path.isdir(true_imgs_path):
                os.makedirs(true_imgs_path)
            
            first_img.save(os.path.join(out_dir, f'first_img.jpg'))
            for i in range(num_imgs_generated):
                this_img = self.mel.audio_slice_to_image(i)
                this_img.save(os.path.join(true_imgs_path, f'true_img_{i}.jpg'))

        # Return tuple of results if return_dict is False
        if not return_dict:
            return out_pil_images, (self.mel.get_sample_rate(), audios)

        return BaseOutput(**AudioPipelineOutput(np.array(audios)[:, np.newaxis, :]), **ImagePipelineOutput(images))


if __name__ == '__main__':
    """
    EXAMPLE USAGE OF CONTINUER PIPELINE
    """
    pipe = ContinuerPipeline.from_pretrained('keonroohparvar/continuer_pipeline')

    pipe(
        audio_path='./results/brian_simpson_saturday_cool.wav',
        out_dir = './test_results',
        )
