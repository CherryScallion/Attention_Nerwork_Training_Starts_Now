#In this script, we will parse the attention data from a text file and convert it into input and output matrices.
#And then we will create a classical attention mechanism function 𝐴(𝑞,𝐾,𝑉) = Σ𝑝(𝑎(𝑘𝑖,𝑞)) × 𝑣𝑖
#And then we train this simple neural network using this attention mechanism.
import numpy as np
import torch
import re
import numpy as np
import os
import torch
import torch.nn as nn
#Let's first parse the data from the text file, which is created by CreateTrainingData.py, that script is 10% ChatGPT4.1 generated, the rest 90% is written by me.
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(script_dir, "TrainingData.txt")
def parse_attention_data(data_string):
    vector_pattern = re.compile(r'\[(-?\d+),\s*(-?\d+)\]')# Regular expression to match vectors like [x, y](this is excusive for this specific dataset we created)
    matches = vector_pattern.findall(data_string)# Find all matches in the input string
    vectors = [list(map(float, match)) for match in matches]# Convert matched strings to float lists
    input_vectors = []
    output_vectors = []
    for i in range(0, len(vectors), 10):
        input_vectors.extend(vectors[i:i+5])
        output_vectors.extend(vectors[i+5:i+10])#A smart way to split the input and output vectors, Grok4 is amazing.
    input_matrices = np.array(input_vectors).reshape(-1, 5, 2)
    output_matrices = np.array(output_vectors).reshape(-1, 5, 2)
    return input_matrices, output_matrices
with open(training_dir, 'r') as file:
    data = file.read()
input_data, output_data = parse_attention_data(data)
# print the shapes and first few matrices to verify
print("Input shapes:", input_data.shape)
print("The first input matrix:\n", input_data[0])
print("\nOut put matrix", output_data.shape)
print("The first output matrix:\n", output_data[0])
# Now, let's define the classical attention mechanism function 𝐴(𝑞,𝐾,𝑉) = Σ 𝑝(𝑎(𝑘𝑖,𝑞)) × 𝑣𝑖, I must mention that I used Claude to learn Attention mechanism.
d_model = 2
# Define Query, Key, Value（d_k, d_v）
# Usually for simple, their dimensions are same to d_model, Gemini told me that.
d_k = 2
d_v = 2
class Classic_Attention_Module(nn.Module):
    def __init__(self, d_model):
        super().__init__()#super function to initialize the parent class nn.Module
        self.d_model = d_model
        self.d_k = d_model  # Usually, d_k is set to be equal to d_model in a simple attention mechanism, Cursor said so.
        # define W_q, W_k, W_v as linear layers
        # input_features: d_model, output_features: d_k/d_v
        self.W_q = nn.Linear(d_model, self.d_k)
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
    def forward(self, q, K, V):
        Q = self.W_q(q)  # (batch_size, seq_len, d_k)
        K = self.W_k(K)  # (batch_size, seq_len, d_k)
        V = self.W_v(V)  # (batch_size, seq_len, d_v
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)) # scores must be torch tensor
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
        output = torch.matmul(attn_weights, V)  # (batch_size, seq_len, d_v)
        return output#Copilot Is All You Need. Only thing you need to do is to push the Tab button and enjoy your 100% AI generated content.
# now, let's define the training process 
def train_attention_model(input_data, output_data, num_epochs=1000, learning_rate=0.01):
    # initalize the model
    model = Classic_Attention_Module(d_model)
    # Define the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    # use Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # transform the numpy input data to torch tensor
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    output_tensor = torch.tensor(output_data, dtype=torch.float32)
    # main training loop
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass: compute the model output
        output_pred = model(input_tensor, input_tensor, input_tensor)  # Using input as K and V as well
        # calculate the loss
        loss = criterion(output_pred, output_tensor)
        # Backpropagation
        loss.backward()
        # Update the model parameters
        optimizer.step()
        # print loss every 100 epochs
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    return model
# Train the model
trained_model = train_attention_model(input_data, output_data)
print("Training complete!")
# Save the trained model
model_path = os.path.join(script_dir, "classic_attention_model.pth")
torch.save(trained_model.state_dict(), model_path)
