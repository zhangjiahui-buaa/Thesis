mkdir output
pip install -r requirements.txt
python train.py --task multi            \
                --multi_type separate   \
                --dataset hateful       \
                --image_enc vit,swint,pit,tnt \
                --text_enc bert         \
