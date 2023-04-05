import torch
from torch import nn
#from torch.optim import adam
from textpreprocessing import X_train, Y_train, all_tags,lem_words,myfile
import random
from torch.utils.data import Dataset, DataLoader
import nltk
from bag_of_words import construct_bag_of_words
from PIL import Image
import sys,time
import textwrap
#from audioInput import getAudio

#data loader class  - load the
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

#Hyperparameters
batch_size = 8 # previously 20
hidden_size = 8 #previously 10
output_size =len(all_tags)
input_size = len(X_train[0])
epochs= 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ChatDataset()
training_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 0)






class ChatBotNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ChatBotNeuralNet,self).__init__()
        #defining nueral network layers
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size,hidden_size)
        self.layer3 = nn.Linear(hidden_size,hidden_size)
        self.layer4 = nn.Linear(hidden_size,hidden_size)
        #self.layer5 = nn.Linear(hidden_size, len(all_tags))
        self.layer5 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.layer4(out)
        out = self.relu(out)
        out = self.layer5(out)

        return out




chatbotmodel = ChatBotNeuralNet(input_size=len(X_train[0]),hidden_size=hidden_size,num_classes=len(all_tags)).to(device)

#loss
lossFunc = nn.CrossEntropyLoss()
model_optimizer = torch.optim.Adam(chatbotmodel.parameters(), lr=0.01) # optimizer for automating the the minimum value of the cost function

for epoch in range(epochs):
  for (words, labels) in training_loader:
    words = words.to(device)
    labels = labels.to(device)
    output = chatbotmodel(words)
    loss = lossFunc(output, labels)

    model_optimizer.zero_grad()
    loss.backward()
    model_optimizer.step()

  if (epoch + 1) % 100 == 0:
    print(f'epoch {epoch + 1 }/{epochs}, loss = {loss.item():.4f}')

print(f'final loss, loss = {loss.item():.4f}')

data = {
    "model_state":chatbotmodel.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "lem_words":lem_words,
    "all_tags":all_tags
}

DATA_FILE = "training_data.pth"
torch.save(data,DATA_FILE)
print("Training Sequence Completed. File saved to {}".format(DATA_FILE))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
lem_words = data['lem_words']
all_tags = data['all_tags']
model_state = data["model_state"]

saved_model = ChatBotNeuralNet(input_size, hidden_size, output_size).to(device)
saved_model.load_state_dict(model_state)
saved_model.eval()



bot_name = "ISABEL-001-0010-0022"
#print("Let's chat! type 'quit' to exit")
print("Welcome to Solent MedBot My name is Isabel, say 'quit to  exit this program")
#myname = input("Please tell us your name: ")

while True:
    # print(myname,": \t")
    # sentence = getAudio()
    # print(sentence)


    sentence = input("User:")
    # sentence = nltk.word_tokenize(sentence)
    if (sentence == "quit"):
        break
    elif (sentence == "show"):
        im = Image.open("mode_of_transmission.jpeg")
        im.show()
    elif (sentence == "yes"):
        myim = Image.open("map.jpeg")
        myim.show()

    # sentence = tokenize(sentence)
    sentence = nltk.word_tokenize(sentence)
    print(sentence)
    # X = bag_of_words(sentence, all_words)
    X = construct_bag_of_words(sentence, lem_words)
    print(X)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = chatbotmodel(X)
    print("Chat Bot Class ###############")
    print(output)
    _, \
    predicted = torch.max(output, dim=1)
    print("Predicted #################################################")
    print(predicted)
    tag = all_tags[predicted.item()]
    print("output for all tags predicted ####################")
    print(tag)
    print("End of  tag ###########")
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.85:
        for intent in myfile['intents']:
            if tag == intent["tag"]:
                print(intent["patterns"])
                print(intent["responses"])
                print("The tag is {} and intent tag is {}".format(tag, intent["tag"]))
                #print(f"{bot_name}: {random.choice(intent['responses'])}")
                print(f"{bot_name}:")
                print("\n")
                print("\n")
                bot_response = random.choice(intent['responses'])
                wrapped_bot_response = textwrap.wrap(bot_response,width=60)
                for line in  wrapped_bot_response:
                    for writer in line:
                        sys.stdout.write(writer)
                        time.sleep(0.1)
                    print()
                #for writer in random.choice(intent['responses']):
                #    sys.stdout.write(writer)
                    #sys.stdout.flush()
                #    time.sleep(0.1)
        print("\n")
        print("\n")
    else:
        print(f"{bot_name}: My knowledge base is finite to selected topics of Covid-19, sorry i can't process all your requests")
        time.sleep(0.1)

