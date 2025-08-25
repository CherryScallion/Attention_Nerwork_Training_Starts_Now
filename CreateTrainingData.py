import os
import numpy as np
import random
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "TrainingData.txt")
def generate_matrix_data(output_path):
    with open(output_path, 'w') as f:
        for i in range(500):
            sum_input_matrix = ""
            sum_output_matrix = ""
            for s in range(1,6):
                input_matrices1 = s*10 + i
                input_matrices2 = i - s*10
                input_matrix = [input_matrices1,input_matrices2]
                output_matrices1 = input_matrices1 + 10
                output_matrices2 = output_matrices1 + input_matrices2
                output_matrix = [output_matrices1,output_matrices2]
                if s == 1:
                    sum_input_matrix = f"{input_matrix}"
                    sum_output_matrix = f"{input_matrix}"
                else:
                    sum_input_matrix = f"{sum_input_matrix},{input_matrix}"
                    sum_output_matrix = f"{sum_output_matrix},{output_matrix}"
            f.write(f" {sum_input_matrix} {sum_output_matrix}\n")
generate_matrix_data(output_path)
#Maybe it was because it was late at night and I was bored, or maybe Plants vs. Zombies had drained my energy, but I felt sleepy as soon as I saw the foreign tadpole characters.
print("Training data created successfully!")

