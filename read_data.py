import ast

data_list = []

with open('my_data.txt', 'r') as f:
    for line in f:
        elements = line.strip().split('+')
        two_d_list = ast.literal_eval(elements[0])
        integer = int(elements[1])
        float_list = [float(i) for i in ast.literal_eval(elements[2])]
        data_list.append([two_d_list, integer, float_list])
        
print(data_list[0][0])