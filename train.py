from preprocess.MVSASingle.dataset import load_MVSA_data_iterator
from transformers import BertTokenizer
import torchvision
from tqdm import tqdm
from model.encoder import CNN_Image_Encoder
from model.decoder import CNN_Image_Decoder
import torch.nn.functional as F
import torch
import torch.optim as optim

train_data_loader, dev_data_loader = load_MVSA_data_iterator('datasets/MVSA_single',
                                      BertTokenizer.from_pretrained("bert-base-uncased"),
                                      torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                                      torchvision.transforms.RandomCrop(224),
                                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                                      torchvision.transforms.ToTensor(),
                                                                      torchvision.transforms.Normalize(
                                                                          [0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])
                                                                      ]),
                                      4,
                                      "cpu",
                                      True,
                                      True,
                                      512)

image_encoder = CNN_Image_Encoder()
image_decoder = CNN_Image_Decoder()
encoder_optimizer = optim.Adam(image_encoder.parameters(), 0.0001)
decoder_optimizer = optim.Adam(image_decoder.parameters(), 0.001)
for epoch in range(10):

    for batch in tqdm(train_data_loader):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        image_input = batch['input_image']
        image_output = image_decoder(image_encoder(image_input))
        prob = F.softmax(image_output, dim=-1)
        gold = batch['image_label'].squeeze(1)
        loss = F.cross_entropy(prob, gold)
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

    total = 0
    correct = 0
    for batch in tqdm(dev_data_loader):
        image_input = batch['input_image']
        image_output = image_decoder(image_encoder(image_input))
        pred = torch.argmax(image_output, dim=-1)
        gold = batch['image_label'].squeeze(1)
        total += len(image_input)
        correct += (pred==gold).sum()
    print("accuracy {}".format(correct/total))
