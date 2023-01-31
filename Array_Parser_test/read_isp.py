from Array_Phaser import Array_Phaser

path = 'chromatix_hi556_j3_main_snapshot_isp.h'
with open(path, 'r', encoding='cp1252') as f:
    text = f.read()

arr_phaser = Array_Phaser(list(text))

abf_node = arr_phaser.get(1).get(3).get(5)
trigger_idx=0

for i in range(len(abf_node.get(trigger_idx).get(0).pos_arr)):
    abf_node.get(trigger_idx).get(0).get(i).text = str(i)

path = 'chromatix_hi556_j3_main_snapshot_isp_modify.h'
with open(path, 'w', encoding='cp1252') as f:
    f.write(''.join(arr_phaser.reconstruct()))


# ABF
# project name: chromatix_hi556_j3_main
# file_path: [project name]\isp\snapshot\[project name]_snapshot.h
# node: [1,3,5]
# trigger_data: 0
# param_data: 1