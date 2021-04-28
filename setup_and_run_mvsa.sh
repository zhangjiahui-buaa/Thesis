mkdir output
pip install -r requirements.txt
pip install git+https://github.com/rwightman/pytorch-image-models.git
python train.py --task image \
                --dataset mvsa \
                --label_num 3   \
                --image_enc vit