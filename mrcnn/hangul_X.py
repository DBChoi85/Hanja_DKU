import os


IMAGE_DIR ='C:\\Users\\ialab\\Desktop\\img_json\\03\\'

file_list = os.listdir(IMAGE_DIR)

'''
for i in file_list:
    i_list = list(i)
    for j in i_list :
        asci_j = ord(j)
        if asci_j >= 48 and asci_j <= 57:
            print("숫자")
        elif asci_j >= 65 and asci_j <=122:
            print("알파벳")
'''
for i in file_list:
    i_list = i
    add_list = []
    print(i_list)
    for j in i_list:
        asci_j = ord(j)
        #print(asci_j)
        if asci_j >= 48 and asci_j <= 57:
            add_list.append(j)
        elif (asci_j >= 65 and asci_j <=90) or \
                (asci_j >= 97 and asci_j <=122):
            add_list.append(j)
        elif asci_j == 95 or asci_j == 46:
            add_list.append(j)
    print(''.join(add_list))
    add_string = ''.join(add_list)
    os.rename(IMAGE_DIR+i_list, IMAGE_DIR+add_string)

#print(ord("a"))
#print(ord("z"))
#print(ord('A'))
#print(ord('Z'))
#print(ord('9'))
#print("aa"+chr(46))
