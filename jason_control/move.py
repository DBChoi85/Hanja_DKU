import os
import json

PATH = "C:\\git\\jason_control\\24700012_a\\"
file_list = os.listdir(PATH)
file_name = file_list[0]
file_stat = os.stat(PATH + file_name)
file_size = file_stat.st_size

file_name2 = file_list[1]
file_stat2 = os.stat(PATH + file_name2)
file_size2 = file_stat2.st_size

json_path = file_list[len(file_list) - 1]
file_path = PATH + json_path

#print(file_size)
a_list = []
b_list = []

with open(file_path, "r") as json_file:
    data = json.load(json_file)
    f_data = data[file_name + str(file_size)]
    regions = f_data['regions']
    loc = regions['0']
    shape = loc['shape_attributes']
    x_points = shape['all_points_x']
    print(x_points)
    print(len(x_points))
    a_list = x_points
    #for idx in regions:
    #    loc = regions[idx]
    #    shape = loc['shape_attributes']
    #    x_points = shape['all_points_x']
    #    print(x_points)

#    print(regions)

with open(file_path, "r") as json_file:
    data = json.load(json_file)
    f_data = data[file_name2 + str(file_size2)]
    regions = f_data['regions']
    loc = regions['20']
    shape = loc['shape_attributes']
    x_points = shape['all_points_x']
    print(x_points)
    print(len(x_points))
    b_list = x_points
    #for idx in regions:
        #loc = regions[idx]
        #shape = loc['shape_attributes']
        #x_points = shape['all_points_x']
        #print(x_points)

print(a_list)
#print(b_list)

d_value = a_list[0] - b_list[0]
print(d_value)

for idx, i in enumerate(b_list):
    b_list[idx] = i + d_value

print(b_list)