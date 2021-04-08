from preprocess.MVSASingle.dataset import load_MVSA_data_iterator
from transformers import BertTokenizer
import torchvision
from tqdm import tqdm
from model.encoder import *
from model.decoder import *
import torch.nn.functional as F
import torch
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
train_data_loader, dev_data_loader = load_MVSA_data_iterator('datasets/MVSA_Single',
                                                             BertTokenizer.from_pretrained("bert-base-uncased"),
                                                             torchvision.transforms.Compose(
                                                                 [torchvision.transforms.Resize(256),
                                                                  torchvision.transforms.CenterCrop(224),
                                                                  # torchvision.transforms.RandomHorizontalFlip(),
                                                                  torchvision.transforms.ToTensor(),
                                                                  torchvision.transforms.Normalize(
                                                                      [0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])
                                                                  ]),
                                                             32,
                                                             device,
                                                             True,
                                                             512)

# image_encoder = ViT_Image_Encoder().to(device)
# image_decoder = Image_Decoder(768).to(device)
classifier = MMBT(3).to(device)
# encoder_optimizer = optim.Adam(image_encoder.parameters(), 0.00003)
# decoder_optimizer = optim.Adam(image_decoder.parameters(), 0.001)
classifier_optimizer = optim.Adam(classifier.parameters(), 0.001)
for epoch in range(10):
    # image_encoder.train()
    # image_decoder.train()
    classifier.train()
    for step, batch in enumerate(train_data_loader):
        # encoder_optimizer.zero_grad()
        # decoder_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        image_input = batch['input_image']
        text_input = batch['input_token_ids']
        label = batch['combined_label'].squeeze(1)
        outputs = classifier.forward(image_input, text_input, labels=label)
        loss, logits = outputs[:2]
        '''image_output = image_decoder(image_encoder(image_input))
        prob = F.softmax(image_output, dim=-1)
        gold = batch['image_label'].squeeze(1)
        loss = F.cross_entropy(prob, gold)'''
        loss.backward()

        # encoder_optimizer.step()
        # decoder_optimizer.step()
        classifier_optimizer.step()
        if step % 10 == 0:
            print(loss)
    with torch.no_grad():
        # image_encoder.eval()
        # image_decoder.eval()
        classifier.eval()
        total = 0
        correct = 0
        for batch in tqdm(dev_data_loader):
            image_input = batch['input_image']
            text_input = batch['input_token_ids']
            label = batch['combined_label'].squeeze(1)
            outputs = classifier(image_input, text_input, labels=label)
            _, logits = outputs[:2]
            #  image_output = image_decoder(image_encoder(image_input))
            # pred = torch.argmax(image_output, dim=-1)
            # gold = batch['image_label'].squeeze(1)
            pred = torch.argmax(logits, dim=-1)
            total += len(image_input)
            correct += (pred == label).sum()
        print("accuracy {}".format(correct / total))
