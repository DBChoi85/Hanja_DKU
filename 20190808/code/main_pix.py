from convert import array_load
from convert2 import array_many_load
from row_segment import row_ch
from col_segment import col_ch

path = 'C:\\Users\\ialab\\Desktop\\Hanja_DKU-master\\sample\\sample01.jpg'
array = array_load(path)


filename = row_ch(array)
print(filename)
#filename = col_ch(array)
filename = array_many_load(filename,'col')
array_many_load(filename,'row',final = True)

'''
filename = col_ch(array)
print(filename)
#filename = col_ch(array)
filename = array_many_load(filename,'row')
array_many_load(filename,'col',final = True)

'''