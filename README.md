# ProcessPainter

Painting process generating using diffusion models

TODO:

- [x] Basic Inference Code Release
- [x] Full Inference Code Release
- [ ] Training Code Release
- [ ] Training Dataset Release
- [x] Checkpoints Release
- [ ] More Examples

ProgressPainter is a plug and play module to generate Human-like painting pregresses.

![main-1](https://p.ipic.vip/sw3mk1.png)

We pretrained the painting module based on multiple traditional painting pregress reconstruction methods.                                                                                                                                                                                                                             

![t2i-1](https://p.ipic.vip/h3zns7.png)

Then we fintuned the painting moodule using very few real-world painting progresses using LoRA (Low Rank Adaptation) techniques, the perfromance of generating painting progresses is amazing.

![lora-1](https://p.ipic.vip/vpuzau.png)

Furthermore, by combining image reference net, we are able to reconstruct existing paintings or finish unfinished painting progresses.

## Inferencing

**MINIMAL 30GB GPU memory is REQUIRED for SINGLE inferencing!!!**

Download the pre-trained models from [Huggingface Repo](https://huggingface.co/nicolaus-huang/ProcessPainter) and then get them into `models` folder correspondingly.

```cmd
wget https://huggingface.co/guoyww/animatediff/blob/main/v3_sd15_adapter.ckpt models/DreamBooth_LoRA/v3_sd15_adapter.ckpt
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

