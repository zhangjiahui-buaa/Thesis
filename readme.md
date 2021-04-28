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

1. Separate(DC, early fusion)

|          | Resnet52 | Vanilla Vit | Vit  | Vanilla Swint | Swint | Vanilla TNT | TNT  | Vanilla PiT | PiT  |
| -------- | -------- | ----------- | ---- | ------------- | ----- | ----------- | ---- | ----------- | ---- |
| Accuracy | 59.8     | 57.4        | 61.2 | 56.60         | 60.8  | 55.40       | 60.4 | 58.00       |      |
| AUROC    |          |             |      | 62.86         |       | 62.82       |      | 63.28       |      |

2. Separate(DC, late fusion)
3. Separate(LTC)
4. Separate(STC)
5. Together(MMBT Style)

|          | Resnet152 | Vit   | Swint | TNT   | PiT   |
| -------- | --------- | ----- | ----- | ----- | ----- |
| Accuracy | 60.60     | 64.80 | 61.40 | 63.80 | 59.40 |
| AUROC    | 65.57     | 69.53 | 67.40 | 66.78 | 66.08 |


### hateful memes unzip password: EWryfbZyNviilcDF

