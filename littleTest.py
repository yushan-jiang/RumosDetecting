import os

src_path = "D:/Bistu2/CourseCode/DataSet/Chinese_Rumor_Dataset-master"
data_root_path = src_path + "/data/"
data_path = os.path.join(data_root_path, 'all_data.txt')

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    # 把数据生成一个元组
for line in lines:
    line = line.replace('\n', '')
    if line == '':
        continue
    content_list = line.split('\t')
    content = content_list[-1]
    if content_list[-1] == '':
        content = content_list[-2]
    print(content)

