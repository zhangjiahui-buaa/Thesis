## Thesis

### prerequisite
`pip install -r requirements.txt`

`pip install git+https://github.com/rwightman/pytorch-image-models.git`

`mkdir output`

### Command
~~~shell script
    python train.py --task {image,text,multi}
                    --multi_type {separate,together}   
                    --dataset {mvsa, hateful}
                    --image_enc {cnn,transformer,vit,swint,pit,tnt}
                    --text_enc {bert}
                    --mixed_enc {mmbt}
                    
~~~

### TODO
- replace the cnn encoder in MMBT with Vision transformer
- fusion technique *important*
  - Concatenate directly(DC)
  - Pass through a linear layer, then concatenate(LTC)
  - Pass through a linear and a self-attention layer, then concatenate(STC)
- for each vit, compare the pretrained version against the vanilla version
- ensemble model
- add visual bert *important!*
- use feature extracted by faster-rcnn
- ablation study
  - 1
  - 2
- error analysis

### Result

#### MVSA

1. Unimodal(Image)

|          | Resnet52 | Vanilla Vit | Vit  | Vanilla Swint | Swint | Vanilla TNT | TNT  | Vanilla PiT | PiT  |
| -------- | -------- | ----------- | ---- | ------------- | ----- | ----------- | ---- | ----------- | ---- |
| Accuracy | 67.50    |             |      |               |       |             |      |             |      |
| AUROC    | 78.89    |             |      |               |       |             |      |             |      |

#### Hateful Meme

1. Separate(DC)

|          | Resnet52 | Vanilla Vit | Vit  | Vanilla Swint | Swint | Vanilla TNT | TNT  | Vanilla PiT | PiT  |
| -------- | -------- | ----------- | ---- | ------------- | ----- | ----------- | ---- | ----------- | ---- |
| Accuracy | 59.8     | 57.4        | 61.2 | 56.60         | 60.8  | 55.40       | 60.4 | 58.00       |      |
| AUROC    |          |             |      | 62.86         |       | 62.82       |      | 63.28       |      |

2. Separate(LTC)
3. Separate(STC)
4. Together(MMBT Style)

|          | Resnet152 | Vit   | Swint | TNT   | PiT  |
| -------- | --------- | ----- | ----- | ----- | ---- |
| Accuracy | 60.60     | 62.40 | 61.40 | 63.80 |      |
| AUROC    | 65.57     | 68.79 | 67.40 | 66.78 |      |


### hateful memes unzip password: EWryfbZyNviilcDF

