#-*-coding:utf-8-*-
import io
import os
import json
import random
import sys

src_path = "D:/Bistu2/CourseCode/DataSet/Chinese_Rumor_Dataset-master"

# 转发与评论信息
rumos_class_dirs = os.listdir(src_path + "/CED_Dataset/rumor-repost/")
non_rumos_class_dirs = os.listdir(src_path + "/CED_Dataset/non-rumor-repost/")

#微博原文路径
original_microblog = src_path + "/CED_Dataset/original-microblog/"

# 谣言标签为0，非谣言标签为1
rumor_label = "0"
non_rumor_label = "1"

# 谣言与非谣言总数
rumor_num = 0
non_rumor_num = 0

all_rumor_list = []
all_non_rumor_list = []


# 生成数据字典
def create_dict(data_path, dict_path):
    with open(dict_path, 'w') as f:
        f.seek(0)
        f.truncate()

    dict_set = set()
    # 读取全部数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 把数据生成一个元组
    for line in lines:
        # 规避掉有的地方有换行，有的地方有sb空格的情况
        line = line.replace('\n', '')
        if line == '':
            continue
        content_list = line.split('\t')
        content = content_list[-1]
        if content_list[-1] == '':
            content = content_list[-2]
        for s in content:
            dict_set.add(s)
    # 把元组转换成字典，一个字对应一个数字
    dict_list = []
    i = 0
    for s in dict_set:
        dict_list.append([s, i])
        i += 1
    # 添加未知字符
    dict_txt = dict(dict_list)
    end_dict = {"<unk>": i}
    dict_txt.update(end_dict)
    end_dict = {"<pad>": i+1}
    dict_txt.update(end_dict)
    # 把这些字典保存到本地中
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(dict_txt))

    print("数据字典生成完成！")


# 创建序列化表示的数据,并按照一定比例划分训练数据train_list.txt与验证数据eval_list.txt
def create_data_list(data_list_path):
    # 在生成数据之前，首先将eval_list.txt和train_list.txt清空
    with open(os.path.join(data_list_path, 'eval_list.txt'), 'w', encoding='utf-8') as f_eval:
        f_eval.seek(0)
        f_eval.truncate()

    with open(os.path.join(data_list_path, 'train_list.txt'), 'w', encoding='utf-8') as f_train:
        f_train.seek(0)
        f_train.truncate()

    with open(os.path.join(data_list_path, 'dict.txt'), 'r', encoding='utf-8') as f_data:
        dict_txt = eval(f_data.readlines()[0])

    with open(os.path.join(data_list_path, 'all_data.txt'), 'r', encoding='utf-8') as f_data:
        lines = f_data.readlines()

    i = 0
    maxlen = 0
    with open(os.path.join(data_list_path, 'eval_list.txt'), 'a', encoding='utf-8') as f_eval, open(
            os.path.join(data_list_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
        for line in lines:
            if line == "\n":
                continue
            words = line.split('\t')[1].replace('\n', '')
            # print(words)
            maxlen = max(maxlen, len(words))
            label = line.split('\t')[0]
            # print(label)
            labs = ""
            # 每8个 抽取一个数据用于验证
            if i % 8 == 0:
                for s in words:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + label + '\n'
                f_eval.write(labs)
            else:
                for s in words:
                    lab = str(dict_txt[s])
                    labs = labs + lab + ','
                labs = labs[:-1]
                labs = labs + '\t' + label + '\n'
                f_train.write(labs)
            i += 1

    print("数据列表生成完成！")
    print("样本最长长度：" + str(maxlen))

# 把生成的数据列表都放在自己的总类别文件夹中
data_root_path = src_path + "/data/"
data_path = os.path.join(data_root_path, 'all_data.txt')
dict_path = os.path.join(data_root_path, "dict.txt")



def load_vocab(file_path):
    fr = open(file_path, 'r', encoding='utf8')
    vocab = eval(fr.read())   #读取的str转换为字典
    fr.close()

    return vocab

def ids_to_str(ids):
    # 返回dict中id对应的str
    words = []
    vocab = load_vocab(os.path.join(data_root_path, 'dict.txt'))
    for k in ids:
        w = list(vocab.keys())[list(vocab.values()).index(int(k))]
        words.append(w if isinstance(w, str) else w.decode('ASCII'))
    return " ".join(words)




if __name__ == '__main__':
    # 解析谣言数据
    for rumos_class_dir in rumos_class_dirs:
        if rumos_class_dir != ".DS_Store" and rumos_class_dir != "._.DS_Store":
            with open(original_microblog + rumos_class_dir, 'r', encoding='UTF-8') as f:
                rumor_content = f.read()
            rumor_dict = json.loads(rumor_content)
            all_rumor_list.append(rumor_label + "\t" + rumor_dict["text"] + "\t")
            rumor_num += 1

    # 解析非谣言数据
    for non_rumos_class_dir in non_rumos_class_dirs:
        if non_rumos_class_dir != ".DS_Store" and non_rumos_class_dir != "._.DS_Store":
            with open(original_microblog + non_rumos_class_dir, 'r', encoding='UTF-8') as f:
                non_rumor_content = f.read()
            non_rumor_dict = json.loads(non_rumor_content)
            all_non_rumor_list.append(non_rumor_label + "\t " + non_rumor_dict["text"].replace("\n", "") + "\n")
            non_rumor_num += 1

    # print("谣言总数为："+ str(rumor_num)) # 1538
    # print("非谣言总数为："+ str(non_rumor_num)) # 1849

    data_list_path = src_path + "/data/"
    all_data_path = data_list_path + "all_data.txt"
    all_data_list = all_rumor_list + all_non_rumor_list
    random.shuffle(all_data_list)

    # 写入前清空
    with open(all_data_path, 'w', encoding='UTF-8') as f:
        f.seek(0)
        f.truncate()

    # 写入data.txt中（a为追加写）
    with open(all_data_path, 'a', encoding='UTF-8') as f:
        for data in all_data_list:
            f.write(data + "\n")

    # 创建数据字典
    create_dict(data_path, dict_path)

    # 创建数据列表
    create_data_list(data_root_path)

    # 打印前2条训练数据
    vocab = load_vocab(os.path.join(data_root_path, 'dict.txt'))

    file_path = os.path.join(data_root_path, 'train_list.txt')
    with io.open(file_path, "r", encoding='utf8') as fin:
        i = 0
        for line in fin:
            i += 1
            cols = line.strip().split("\t")
            if len(cols) != 2:
                sys.stderr.write("[NOTICE] Error Format Line!")
                continue
            label = int(cols[1])
            wids = cols[0].split(",")
            print(str(i) + ":")
            print('sentence list id is:', wids)
            print('sentence list is: ', ids_to_str(wids))
            print('sentence label id is:', label)
            print('---------------------------------')

            if i == 2: break



