from Array_Parser import Array_Parser

path = 'chromatix_hi556_j3_main_snapshot_cpp.h'
with open(path, 'r', encoding='cp1252') as f:
    text = f.read()

arr_phaser = Array_Parser(list(text))
trigger_idx=0

##### ASF #####
asf_node = arr_phaser.get(1).get(3)
print(''.join(asf_node.get(trigger_idx).get(9).reconstruct()))

# for i in range(len(asf_node.get(trigger_idx).get(0).pos_arr)):
#     # asf_node.get(trigger_idx).get(1).get(i).text = str(i)
#     print(''.join(asf_node.get(trigger_idx).get(9).reconstruct()))



# print(''.join(asf_node.get(trigger_idx).get(0).reconstruct()))
# path = 'chromatix_hi556_j3_main_snapshot_cpp_modify.h'
# with open(path, 'w', encoding='cp1252') as f:
#     f.write(''.join(arr_phaser.reconstruct()))

# ASF
# project name: chromatix_hi556_j3_main
# file_path: [project name]\cpp\cpp_snapshot\[project name]_snapshot_cpp.h
# node: [1,3]
# trigger_data: 0
# param_data: 1

##### ASF #####

##### WNR #####
# asf_node = arr_phaser.get(4).get(3)
# print(''.join(asf_node.get(trigger_idx).get(1).reconstruct()))

##### WNR #####
