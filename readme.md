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
- fusion technique
- for each vit, compare the pretrained version against the vanilla version
- ensemble model
- add visual bert *important!*
- use feature extracted by faster-rcnn

### Result

1. Separate(Simple concatenate)

|          | Resnet52 | Vanilla Vit | Vit  | Vanilla Swint | Swint | Vanilla TNT | TNT  | Vanilla PiT | PiT  |
| -------- | -------- | ----------- | ---- | ------------- | ----- | ----------- | ---- | ----------- | ---- |
| Accuracy | 59.8     | 57.4        | 61.2 |               | 60.8  |             | 60.4 |             |      |
| AUROC    |          |             |      |               |       |             |      |             |      |

2. Together(MMBT Style)

|          | Resnet152 | Vanilla Vit | Vit  | Vanilla Swint | Swint | Vanilla TNT | TNT  | Vanilla PiT | PiT  |
| -------- | --------- | ----------- | ---- | ------------- | ----- | ----------- | ---- | ----------- | ---- |
| Accuracy | 62.8      | 57.4        | 61.2 |               | 60.8  |             | 60.4 |             |      |
| AUROC    | 64.2      |             |      |               |       |             |      |             |      |


### hateful memes unzip password: EWryfbZyNviilcDF

