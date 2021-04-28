mkdir output
pip install -r requirements.txt
pip install git+https://github.com/rwightman/pytorch-image-models.git
python train.py --task multi            \
                --multi_type separate   \
                --dataset hateful       \
                --image_enc vit,swint,pit,tnt \
                --text_enc bert         \
