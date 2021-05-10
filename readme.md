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
- replace the cnn encoder in MMBT with Vision transformer(Done)
- fusion technique *important*
  - Concatenate directly(DC)
  - Pass through a linear layer, then concatenate(LTC)
  - Pass through a linear and a self-attention layer, then concatenate(STC)
- for each vit, compare the pretrained version against the vanilla version(Done)
- ensemble model(Done)
- use feature extracted by faster-rcnn(Use the figure provided by Facebook)
- ablation study
  - 1
  - 2
- error analysis(complete in this weekend)
- deploy model(use streamlit, complete in this weekend, Done)

### Result

#### MVSA

1. Unimodal(Image)

|          | Vanilla Resnet52 | Resnet52 | Vanilla Vit | Vit   | Vanilla Swint | Swint     | Vanilla TNT | TNT   | Vanilla PiT | PiT   |
| -------- | ---------------- | -------- | ----------- | ----- | ------------- | --------- | ----------- | ----- | ----------- | ----- |
| Accuracy | 56.00            | 67.50    | 58.75       | 66.75 | 59.25         | 67.75     | 58.75       | 66.75 | 58.50       | 66.00 |
| AUROC    | **66.21**        | 78.89    | 65.45       | 81.19 | 61.02         | **81.79** | 64.61       | 78.94 | 62.85       | 80.42 |

#### Hateful Meme

1. Separate(DC)

|          | Resnet152 | Vanilla Vit | Vit   | Vanilla Swint | Swint | Vanilla TNT | TNT   | Vanilla PiT | PiT   |
| -------- | --------- | ----------- | ----- | ------------- | ----- | ----------- | ----- | ----------- | ----- |
| Accuracy | 60.40     | 57.4        | 61.60 | 56.60         | 60.00 | 55.40       | 59.40 | 58.00       | 60.60 |
| AUROC    | 62.90     | 62.78       | 66.80 | 62.86         | 66.20 | 62.82       | 65.37 | 63.28       | 65.66 |

2. Ensemble Separate

   |          | Resnet+Vit+Swint | Vit+Swint+TNT | Resnet+Vit+Swint+Tnt+Pit |
   | -------- | ---------------- | ------------- | ------------------------ |
   | Accuracy | 62.40            | 59.20         | 60.20                    |
   | AUROC    | 67.11            | 65.49         | 66.25                    |

    

3. Separate(LTC)

4. Separate(STC)

5. Together(MMBT Style)

|          | Resnet152 | Vit   | Swint | TNT   | PiT   |
| -------- | --------- | ----- | ----- | ----- | ----- |
| Accuracy | 60.60     | 62.40 | 61.40 | 63.80 | 62.60 |
| AUROC    | 65.57     | 68.79 | 67.40 | 66.78 | 66.92 |

### hateful memes unzip password: EWryfbZyNviilcDF

