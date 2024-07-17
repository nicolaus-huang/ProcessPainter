# ProcessPainter

Painting process generating using diffusion models

TODO:

- [x] Basic Inference Code Release
- [ ] Full Inference Code Release
- [ ] Training Code Release
- [ ] Training Dataset Release
- [x] Checkpoints Release

ProgressPainter is a plug and play module to generate Human-like painting pregresses.

![main-1](https://p.ipic.vip/sw3mk1.png)

We pretrained the painting module based on multiple traditional painting pregress reconstruction methods.                                                                                                                                                                                                                             

![t2i-1](https://p.ipic.vip/h3zns7.png)

Then we fintuned the painting moodule using very few real-world painting progresses using LoRA (Low Rank Adaptation) techniques, the perfromance of generating painting progresses is amazing.

![lora-1](https://p.ipic.vip/vpuzau.png)

Furthermore, by combining image reference net, we are able to reconstruct existing paintings or fish unfinished painting progresses.

## Inferencing

Download the pre-trained models from [Google Drive](https://drive.google.com/file/d/1Kgr2uieeY5GJr0TtQJkHt7sQw1PMEstj/view?usp=share_link) and then unzip them into `models` folder.

##### Painting Characters

```cmd
python scripts/animate.py --config configs/prompts/character.yaml 
```

##### Painting Buildings

```cmd
python scripts/animate.py --config configs/prompts/speedpainting.yaml 
```

