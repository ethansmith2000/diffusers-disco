import inspect
from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from transformers import CLIPImageProcessor, CLIPModel, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput

from losses import spherical_dist_loss, symm_loss, tv_loss, range_loss, MakeCutoutsES, MakeCutouts



def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value



class CLIPGuidedStableDiffusion(DiffusionPipeline):
    """CLIP guided stable diffusion based on the amazing repo by @crowsonkb and @Jack000
    - https://github.com/Jack000/glid-3-xl
    - https://github.dev/crowsonkb/k-diffusion
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler,DDPMScheduler, DPMSolverMultistepScheduler],
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

        self.normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        set_requires_grad(self.text_encoder, False)
        set_requires_grad(self.unet, False)
        set_requires_grad(self.vae, False)

        self.unet.enable_gradient_checkpointing()
        self.vae.enable_gradient_checkpointing()

    def denoise(self, latents, noise_pred, timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        return pred_original_sample, beta_prod_t

    def add_clip_models(self, clip_models, clip_sizes, clip_tokenizers):
        self.clip_models = clip_models
        self.clip_sizes = clip_sizes
        self.clip_tokenizers = clip_tokenizers

        for clip_model in self.clip_models:
            set_requires_grad(clip_model, False)
            if clip_model.__class__.__name__ == "CLIPModel":
                clip_model.gradient_checkpointing_enable()
            else:
                clip_model.set_grad_checkpointing(enabled=True)

    def decode_into_image(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        return image

    @torch.enable_grad()
    def cond_fn(
        self,
        latents,
        timestep,
        guidance_scale,
        text_embeddings,
        all_text_embeddings_clip,
        clip_guidance_scale,
        clamp_value,
        use_predx0,
        optimization_steps,

        num_overview,
        num_inner,
        cut_power,
        innercut_gray_p,
        overview_gray_p,
        overview_proportion,
        floor,
        all_models_same_cuts,

        col_jitter,
        pixel_jitter,
        percent_stop_affine,
        degree_tilt,
        affine,
        hflip,
        perspective,

        adaptive_weight,
        overview_type,
        skip_augs,
        resize_type
    ):

        total_cuts = num_overview + num_inner

        for i in range(optimization_steps):
            # 
            latents = latents.detach().requires_grad_()
            latent_model_input = self.scheduler.scale_model_input(latents, timestep)
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latent_model_input] * 2)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # get x0
            pred_original_sample, beta_prod_t = self.denoise(latents, noise_pred, timestep)

            if use_predx0:
                sample = pred_original_sample
            else:
                # get xt-1
                fac = torch.sqrt(beta_prod_t)
                sample = pred_original_sample * (fac) + latents * (1 - fac)

            # decode into pixels
            image = self.decode_into_image(pred_original_sample)

            # make cutouts and norm for clip
            loss = torch.zeros(1, device=self.device)

            if all_models_same_cuts:
                randoffsetx = torch.rand(total_cuts)
                randoffsety = torch.rand(total_cuts)
                randsize = torch.rand(total_cuts)
                randsize2 = torch.rand(total_cuts)
            else:
                randoffsetx = None
                randoffsety = None
                randsize = None
                randsize2 = None

            for i in range(len(self.clip_models)):
                if num_overview > 0 or num_inner > 0:
                    # tmstep = 1 - (timestep / 1000)
                    # make_cutouts = MakeCutoutsES(
                    #     cut_size = self.clip_sizes[i],
                    #     randoffsetx=randoffsetx,
                    #     randoffsety=randoffsety,
                    #     randsize=randsize,
                    #     randsize2=randsize2,
                    #     timestep=tmstep,
                    #     Overview = num_overview,
                    #     InnerCrop = num_inner,
                    #     IC_Size_Pow=cut_power,
                    #     IC_Grey = innercut_gray_p,
                    #     overview_gray = overview_gray_p,
                    #     ov_proportion = overview_proportion,
                    #     floor = floor,
                    #     all_models_same_cuts = False,

                    #     col_jitter = col_jitter,
                    #         pixel_jitter = pixel_jitter,
                    #     percent_stop_affine = percent_stop_affine,
                    #         degree_tilt = degree_tilt,
                    #         affine = affine,
                    #         hflip = hflip,
                    #         perspective = perspective,

                    #     adaptive_weight = adaptive_weight,
                    #     overview_type = overview_type,
                    #     skip_augs=skip_augs,
                    #     resize_type = resize_type
                    # )
                    # image = make_cutouts(image)
                    make_cutouts = MakeCutouts()
                    image = make_cutouts(image, total_cuts, self.clip_sizes[i], cut_power=cut_power)
                image = self.normalize(image).to(latents.dtype)

                # encode with clips
                if self.clip_models[i].__class__.__name__ == "CLIPModel":
                    image_embeddings_clip = self.clip_models[i].get_image_features(image)
                else:
                    image_embeddings_clip = self.clip_models[i].encode_image(image)
                image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

                # get clip losses
                dists = spherical_dist_loss(image_embeddings_clip, all_text_embeddings_clip[i])
                dists = dists.view([total_cuts, sample.shape[0], -1])
                clip_loss = dists.sum(2).mean(0).sum() * clip_guidance_scale

                loss += clip_loss

            loss = loss / len(self.clip_models)

            # modify noise pred based on our gradients
            grads = -torch.autograd.grad(loss, latents)[0]
            grads = torch.clamp(grads, -clamp_value, clamp_value)
            # noise_pred = noise_pred - torch.sqrt(beta_prod_t) * grads
            latents = latents - grads

        return noise_pred, latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = "",
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 2,
        num_optimization_steps = 20,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        clip_guidance_scale: Optional[float] = 100,
        clip_prompt: Optional[Union[str, List[str]]] = None,
        clamp_value = 0.25,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        use_predx0 = True,
        optimization_steps=10,

        num_overview = 4,
        num_inner = 12,
        cut_power = 1.0,
        innercut_gray_p = 0.1,
        overview_gray_p = 0.1,
        overview_proportion = 0.5,
        floor = 0.1,
        all_models_same_cuts = False,

        col_jitter = 0.1,
        pixel_jitter = 0.1,
        percent_stop_affine = 0.5,
        degree_tilt = 0.1,
        affine = True,
        hflip = True,
        perspective = True,

        adaptive_weight = True,
        overview_type = "jaxy_overviews",
        skip_augs = False,
        resize_type = "bilinear",
    ):
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # get prompt text embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        # duplicate text embeddings for each generation per prompt
        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        # get clip prompt text embeddings
        all_text_embeddings_clip = []
        for i in range(len(self.clip_models)):
            if clip_guidance_scale > 0:
                if self.clip_models[i].__class__.__name__ == "CLIPModel":
                    is_hf = True
                else:
                    is_hf = False

                if is_hf:
                    clip_text_input = self.clip_tokenizers[i](
                        clip_prompt,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(self.device)
                    text_embeddings_clip = self.clip_models[i].get_text_features(clip_text_input)
                else:
                    clip_text_input = self.clip_tokenizers[i](clip_prompt, context_length=77)
                    text_embeddings_clip = self.clip_models[i].encode_text(clip_text_input)

                text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
                # duplicate text embeddings clip for each generation per prompt
                text_embeddings_clip = text_embeddings_clip.repeat_interleave(num_images_per_prompt, dim=0)

            all_text_embeddings_clip.append(text_embeddings_clip)

        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            max_length = text_input.input_ids.shape[-1]
            uncond_input = self.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            # duplicate unconditional embeddings for each generation per prompt
            uncond_embeddings = uncond_embeddings.repeat_interleave(num_images_per_prompt, dim=0)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # get the initial random noise unless the user supplied it
        latents_shape = (batch_size * num_images_per_prompt, self.unet.config.in_channels, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if latents is None:
            latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        latents = latents.to(self.device)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Some schedulers like PNDM have timesteps as arrays
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # # expand the latents if we are doing classifier free guidance
            # latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # # predict the noise residual
            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # # perform classifier free guidance
            # if do_classifier_free_guidance:
            #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # perform clip guidance
            if clip_guidance_scale > 0:
                noise_pred, latents = self.cond_fn(
                    latents,
                    t,
                    guidance_scale,
                    text_embeddings,
                    all_text_embeddings_clip,
                    clip_guidance_scale,
                    clamp_value,
                    use_predx0,
                    optimization_steps=optimization_steps,

                    num_overview = num_overview,
                    num_inner = num_inner,
                    cut_power = cut_power,
                    innercut_gray_p = innercut_gray_p,
                    overview_gray_p = overview_gray_p,
                    overview_proportion = overview_proportion,
                    floor = floor,
                    all_models_same_cuts = all_models_same_cuts,

                    col_jitter=col_jitter,
                    pixel_jitter=pixel_jitter,
                    percent_stop_affine=percent_stop_affine,
                    degree_tilt=degree_tilt,
                    affine=affine,
                    hflip=hflip,
                    perspective=perspective,

                    adaptive_weight=adaptive_weight,
                    overview_type=overview_type,
                    skip_augs=skip_augs,
                    resize_type=resize_type,
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        image = self.decode_into_image(latents)
        image = image.float().cpu().permute(0, 2, 3, 1).numpy()

        image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)
