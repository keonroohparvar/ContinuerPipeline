# Adapted from https://github.com/teticio/audio-diffusion/blob/main/audiodiffusion/pipeline_audio_diffusion.py

from typing import Optional, Tuple, Union
import os

from PIL import Image
import PIL
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
        out_dir: Optional[str] = None,
        name: Optional[str] = None
    ) -> Image.Image :
        if diffusion_pipeline_name is None:
            DEFAULT_AUDIO_DIFFUSION_GENERATOR = 'teticio/latent-audio-diffusion-256'
            pipe = DiffusionPipeline.from_pretrained(DEFAULT_AUDIO_DIFFUSION_GENERATOR).to(self.device)
            print('using default pipe!')
        else:
            pipe = DiffusionPipeline.from_pretrained(diffusion_pipeline_name).to(self.device)
        
        seed = generator.seed()
        print(f'Seed = {seed}')
        generator.manual_seed(seed)

        pipe_output = pipe(generator=generator)
        # pipe_output = pipe(
        #      steps=50,
        #     generator=generator
        # )
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

        if out_dir is not None:
            img_output.save(os.path.join(out_dir, name))

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
        print(f'Training Seed: {gen_seed}')
        generator.manual_seed(gen_seed)

        # Initialize step generator if not passed in
        step_generator = step_generator or generator

        first_img = None

        # Get root image from either raw audio or input image
        if audio_path is not None or raw_audio is not None:
            self.mel.load_audio(audio_path, raw_audio)
            root_audio = self.mel.audio
            number_of_slices = self.mel.get_number_of_slices()
            print(f'Number of slices in audio: {number_of_slices}')
            first_img = self.mel.audio_slice_to_image(slice=0)
            prev_img = self.mel.audio_slice_to_image(slice=(number_of_slices-1))
        
        else:
            print(f'Error. Did not provide audio_path or raw_audio arguments.')
            exit()

        
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

        print(f'Min | Max of Root: {torch.min(root_latents)} {torch.max(root_latents)}')
        print(f'Min | Max of prev: {torch.min(prev_latents)} {torch.max(prev_latents)}')
        print(f'Min | Max of prev: {torch.min(target_latents)} {torch.max(target_latents)}')
        print(f'Averages of root, prev, latents: {torch.mean(root_latents)} | {torch.mean(prev_latents)} | {torch.mean(target_latents)}')
        print(f'shapes of root, prev, latents: {root_latents.shape} | {prev_latents.shape} | {target_latents.shape}')


            
       # Initialize the generator to a printed seed
        step_seed = step_generator.seed()
        print(f'Training Seed: {step_seed}')
        step_generator.manual_seed(step_seed)

        # Create array for output images
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
            # latents = latents * 0.18215

            # Do backwards diffusion
            for step, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                # Configure input for model
                model_input = torch.cat([root_latents, prev_latents, target_latents], axis=1)
                if step == 0:
                    print(f'Model input shape: {model_input.shape}')


                if step % 100 == 0:
                    print(f'Min|Max|Mean of target latents: {torch.min(target_latents)} | {torch.max(target_latents)} | {torch.mean(target_latents)}')

                model_output = self.unet(model_input, t)["sample"]

                if step == 0:
                    print('output!')
                    print(model_output)

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
                

            prev_latents = target_latents.detach().clone()
            if self.vqvae is not None:
                # 0.18215 was scaling factor used in training to ensure unit variance
                target_latents = 1 / 0.18215 * target_latents
                images = self.vqvae.decode(target_latents)["sample"]
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
            # audios.insert(0, root_audio)
            audio_concatenated = np.concatenate(audios)
            self.export_audio(audio_concatenated, os.path.join(out_dir,'entire_song.wav'))

            # For debugging purposes
            if os.getenv('DEBUG_CONTINUER', 'false').lower() in ('true', '1', 't'):
                true_imgs_path = os.path.join(out_dir, 'true_imgs')
                if not os.path.isdir(true_imgs_path):
                    os.makedirs(true_imgs_path)
                
                first_img.save(os.path.join(out_dir, f'first_img.jpg'))
                for i in range(num_imgs_generated):
                    this_img = self.mel.audio_slice_to_image(i)
                    this_img.save(os.path.join(true_imgs_path, f'true_img_{i}.jpg'))



        if not return_dict:
            return images, (self.mel.get_sample_rate(), audios)

        return BaseOutput(**AudioPipelineOutput(np.array(audios)[:, np.newaxis, :]), **ImagePipelineOutput(images))


if __name__ == '__main__':
    model_path = '../keons_continuer'

    unet = UNet2DModel.from_pretrained(model_path, subfolder='unet')
    vqvae = AutoencoderKL.from_pretrained(model_path, subfolder='vqvae')
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder='scheduler')
    mel = Mel.from_pretrained(model_path, subfolder='mel')

    pipe = ContinuerPipeline(
        unet = unet,
        scheduler = scheduler,
        vqvae = vqvae,
        mel = mel
    )

    # pipe(
    #     # root_img=Image.open('root_img.jpg'),
    #     num_imgs_generated=5,
    #     out_dir='keon_results'
    # )

    wav_file = 'results/train_dataset_imgs/root_wav.wav'
    pipe(
        audio_path = wav_file,
        num_imgs_generated = 2,
        out_dir = 'results/results_on_train_dataset'
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



