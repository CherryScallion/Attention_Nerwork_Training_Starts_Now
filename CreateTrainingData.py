import os
import numpy as np
import random
import torch
#script_dir = os.path.dirname(os.path.abspath(__file__))
#output_path = os.path.join(script_dir, "TrainingData.txt")
#def generate_matrix_data(output_path):
#    with open(output_path, 'w') as f:
#        for i in range(500):
#            sum_input_matrix = ""
#            sum_output_matrix = ""
#            for s in range(1,6):
#                input_matrices1 = s*10 + i
#                input_matrices2 = i - s*10
#                input_matrix = [input_matrices1,input_matrices2]
#                output_matrices1 = input_matrices1 + 10
#                output_matrices2 = output_matrices1 + input_matrices2
#                output_matrix = [output_matrices1,output_matrices2]
#                if s == 1:
#                    sum_input_matrix = f"{input_matrix}"
#                    sum_output_matrix = f"{input_matrix}"
#                else:
#                    sum_input_matrix = f"{sum_input_matrix},{input_matrix}"
#                    sum_output_matrix = f"{sum_output_matrix},{output_matrix}"
#            f.write(f" {sum_input_matrix} {sum_output_matrix}\n")
#generate_matrix_data(output_path)
#Maybe it was because it was late at night and I was bored, or maybe Plants vs. Zombies had drained my energy, but I felt sleepy as soon as I saw the foreign tadpole characters.
#print("Training data created successfully!")
#The old script above is totally useless, the model CANNOT learn anything from it. I keep it as a warning. Use the code below.

# Define the script's directory for file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
training_path = os.path.join(script_dir, "TrainingData.txt")
validation_path = os.path.join(script_dir, "ValidationData.txt")
# Set global parameters for the data generation process
num_train_samples = 5000
num_val_samples = 1000
seq_len = 5  # Sequence length, i.e., the number of vectors in a sequence
d_model = 2  # The dimension of each vector (e.g., [x, y])
d_k = d_v = 2  # The dimensions of Query, Key, and Value vectors
# A fixed random seed ensures the data is reproducible every time
np.random.seed(39)
def generate_hybrid_attention_data(num_samples, output_path):
    """
    Generates training or validation data in a format compatible with the user's
    parsing function, while correctly simulating an attention mechanism.
    
    The script uses simple, non-random weight matrices to make the transformation
    from input to output more predictable and understandable for debugging.

    Args:
        num_samples (int): The number of data sequences to generate.
        output_path (str): The path to the output file.
    """
    with open(output_path, 'w') as f:
        # Define simple, fixed weight matrices to make the transformation transparent
        W_q = np.array([[1.5, 2.0], [2.0, 1.5]])  # Identity matrix
        W_k = np.array([[1.5, 2.0], [2.0, 1.5]])  # Identity matrix
        W_v = np.array([[1.5, 2.0], [2.0, 1.5]])  # Identity matrix
        for i in range(num_samples):
            # Generate a simple, predictable input sequence
            # For example, the first group is [0, 1], [1, 2]...
            # The second group is [1, 2], [2, 3]...
            # This makes it easier to debug and understand the transformation
            input_vectors = np.array([[i + j, i + j + 1] for j in range(seq_len)])
            # Convert to float to ensure compatibility with PyTorch training later
            input_vectors = input_vectors.astype(np.float32)
            #This part is essiential,it simulates the attention mechanism correctly            
            #Compute Q, K, V by multiplying the input vectors with fixed weight matrices
            Q = np.dot(input_vectors, W_q)
            K = np.dot(input_vectors, W_k)
            V = np.dot(input_vectors, W_v)
            #Calculate attention scores (dot product between Q and K.T)
            #calculate scaled dot-product attention
            scores = np.dot(Q, K.T)/np.sqrt(d_k)
            #Apply softmax to get attention weights
            #softmax along the last axis (seq_len)
            scores = torch.tensor(scores, dtype=torch.float32)
            attn_weights = torch.softmax(scores, dim=-1).numpy()
            # Compute the final output (weighted sum of V)
            # compute the final output as the weighted sum of V
            output_vectorss = np.dot(attn_weights, V)
            output_vectors = output_vectorss.astype(np.float32)            
            # This part ensures the data can be parsed by your existing function            
            # Format the input and output vectors into a single string
            line_parts = []            
            # Add input vectors to the string
            for vec in input_vectors:
                line_parts.append(f"[{vec[0]:.2f}, {vec[1]:.2f}]")            
            # Add output vectors to the string
            for vec in output_vectors:
                line_parts.append(f"[{vec[0]:.2f}, {vec[1]:.2f}]")            
            # Write the final line to the file, separated by spaces
            f.write(" ".join(line_parts) + "\n")
    print(f"'{os.path.basename(output_path)}' created successfully with {num_samples} samples!")
# Generate training data and validation data
generate_hybrid_attention_data(num_train_samples, training_path)
generate_hybrid_attention_data(num_val_samples, validation_path)



