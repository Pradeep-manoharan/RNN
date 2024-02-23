# Library

import os
import numpy as np
import unicodedata
import string

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Hyper-parameters
num_epochs = 2



# Data Processing
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
print(n_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != "Mn"
        and c in all_letters
    )

all_name = []
all_country = []

for f in os.listdir("data/data/names"):
    with open("data/data/names/"+f,'r',encoding = "utf-8") as f1:
        lis = f1.readlines()
        clean_list = list(map(unicodeToAscii,lis))
        all_name.extend(clean_list)
        all_country.extend([f.split(".")[0]] * len(clean_list))

n_rows = len(all_name)



emb = torch.eye(n_letters)
mapping = dict(zip(np.unique(all_country),range(n_rows)))




def get_data(idx):
    name = all_name[idx]
    country = all_country[idx]
    name_char_lis = np.array(list(name))
    indices = np.where(name_char_lis[... , None] == np.array(list(all_letters)))[1]
    data = emb[torch.from_numpy(indices)].unsqueeze(0)
    target = torch.tensor([mapping[country]])
    return data,target


# Model Building

class RNN(nn.Module):
    def __init__(self,n_country,n_letters):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(n_letters, 2 * n_letters,batch_first=True)
        self.fc = nn.Linear(2*n_letters,n_country)
    def forward(self,x):
        out, _ = self.rnn(x)
        out= self.fc(out[:,-1,:])
        return out


model = RNN(len(np.unique(all_country)),n_letters)
print(model)


criteria =  nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

all_losses = []
for epoch in range(num_epochs):
    arr = np.arange(n_rows)
    np.random.shuffle(arr)
    epoch_loss = 0
    for ind in arr:
        data,target  = get_data(ind)
        output = model(data)
        loss = criteria(output,target)
        epoch_loss+=loss.detach().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (ind+1) % 1000== 0:
            print(f'Epoch [{epoch+1}/{num_epochs}],Loss [{loss}]')
    all_losses.append(epoch_loss)

plt.plot(all_losses)
plt.show()