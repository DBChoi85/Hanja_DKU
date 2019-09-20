
import os
import json
import numpy as np


PATH = 'C:\\Users\\ialab\\Desktop\\img_json\\'
#PATH = "C:\\Users\\ialab\\Desktop\\Hanja_DKU\\jason_control\\mm_a_001_387b\\"
file_list = os.listdir(PATH)
file_name = file_list[0]
print('file_name',file_name)
file_stat = os.stat(PATH + file_name)
file_size = file_stat.st_size

count = 0
json_mask = dict()
M_regions = dict()
M_name = dict()

for top_i in range(1,len(file_list)):
    file_name2 = file_list[top_i]
    file_name_spl = file_name2.split(".")
    # print(file_name_spl[1])
    if file_name_spl[1] == "json": break
    print('file_name2',file_name2)
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

        filename_join = file_name + str(file_size)


        regions = f_data['regions']
        #for dex in range(len(regions)):

        loc = regions['%d' % (top_i - 1)]
        shape = loc['shape_attributes']
        x_points = shape['all_points_x']
        # print(x_points)
        # print(len(x_points))
        a_list = x_points
        # print('a_list',a_list)
        # for idx in regions:
        #    loc = regions[idx]
        #    shape = loc['shape_attributes']
        #    x_points = shape['all_points_x']
        #    print(x_points)

        f_data2 = data[file_name2 + str(file_size2)]
        regions2 = f_data2['regions']
        # print(len(regions2))
        for ax in range(len(regions2)):
            n_b_list = []
            loc2 = regions2['%d' % ax]
            shape2 = loc2['shape_attributes']
            x_points = shape2['all_points_x']
            # print(x_points)
            # print(len(x_points))
            b_list = x_points
            # print('b_list',b_list)
            d_value = a_list[0] - b_list[0]
            for i in b_list:
                n_b_list.append(i + d_value)
            # print(n_b_list)
            M_shape = dict()
            M_shape["name"] = "polygon"
            M_shape["all_points_x"] = n_b_list
            M_shape["all_points_y"] = shape2['all_points_y']
            # print(n_b_list)

            M_re_attrib = dict()
            M_re_attrib["name"] = "hanja"

            M_num = dict()
            M_num["shape_attributes"] = M_shape
            M_num["region_attributes"] = M_re_attrib

            M_regions["%d" % count] = M_num
            count += 1

        M_name["fileref"] = ""
        M_name["size"] = file_size
        M_name["filename"] = file_name
        M_name["base64_img_data"] = ""
        M_name["file_attributes"] = {}
        M_name["regions"] = M_regions

        json_mask[filename_join] = M_name

    make_file = open(PATH + 'test.json', 'w', encoding="utf-8")
    json.dump(json_mask, make_file, ensure_ascii=False, indent="\t")

#    print(regions)


#print(a_list)
#print(b_list)

