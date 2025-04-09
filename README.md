# ProcessPainter


<a href="https://arxiv.org/abs/2406.06062"><img src="https://img.shields.io/badge/ariXv-2406.06062-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/nicolaus-huang/ProcessPainter/tree/main"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace"></a>


Painting process generating using diffusion models

TODO:

- [x] Basic Inference Code Release
- [x] Full Inference Code Release
- [x] Training Code Release
- [x] Training Dataset Release
- [x] Checkpoints Release

ProgressPainter is a model based on animatediff to generate Human-like painting progresses.

![main-1](https://p.ipic.vip/sw3mk1.png)

We pretrained the painting module based on multiple traditional painting pregress reconstruction methods.                                                                                                                                                                                                                             

![t2i-1](https://p.ipic.vip/h3zns7.png)

Then we fintuned the painting moodule using very few real-world painting progresses with LoRA (Low Rank Adaptation) techniques, the perfromance of generating painting progresses is amazing.

![lora-1](https://s2.loli.net/2024/07/23/J2qGgEtz8ZPrmAN.jpg)

Furthermore, by combining image reference net, we are able to reconstruct existing paintings or finish unfinished painting progresses.

## Inferencing

**MINIMAL 30GB GPU memory is REQUIRED for SINGLE inferencing!!!**

Download the pre-trained models from [Huggingface Repo](https://huggingface.co/nicolaus-huang/ProcessPainter) and then get them into `models` folder accordingly. If you need to do speedpaintings, you would need to download the dreambooth model from the community [HERE](https://comfy.icu/files/revAnimated_v2Rebirth.safetensors). 

```cmd
wget -O models/DreamBooth_LoRA/v3_sd15_adapter.ckpt https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_adapter.ckpt
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 models/stable-diffusion-v1-5
```
##### Painting Characters

```cmd
python scripts/animate.py --config configs/prompts/character.yaml 
```

##### Painting Buildings

```cmd
python scripts/animate.py --config configs/prompts/speedpainting.yaml 
```

## Acknowledgment

[Stylized Neural Painting](https://jiupinjia.github.io/neuralpainter/)

[Learning to Paint With Model-based Deep Reinforcement Learning](https://github.com/hzwer/ICCV2019-LearningToPaint)

[Paint Transformer: Feed Forward Neural Painting with Stroke Prediction](https://github.com/Huage001/PaintTransformer)

[AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://github.com/guoyww/animatediff/)

Training data comes from @"Doodling by the Lakeside.", thanks for his contribution.
