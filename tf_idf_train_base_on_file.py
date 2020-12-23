"""
    date:       2020/12/22 8:08 下午
    written by: neonleexiang
"""
import os
import json
import jieba
import jieba.posseg as pseg

import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle   # we use pickle to save our model file


# # 获取文件列表
def get_file_list(path):
    file_list = []
    files = os.listdir(path)
    for f in files:
        if f[0] == '.':
            pass
        else:
            # filelist.append(os.path.join(path, f))
            file_list.append(f)
    return file_list


# # 对文档进行分词处理
# def segFile(segFileName, recordPath = 'segFile', filePath = 'corpus'):
#     # 保存分词结果的目录
#     if not os.path.exists(recordPath):
#         os.mkdir(recordPath)
#     # 读取文档
#     segFilePath = os.path.join(filePath, segFileName)
#     fileObj = open(segFilePath, 'r+')
#     fileData = fileObj.read()
#     fileObj.close()
#
#     # 对文档进行分词处理，采用默认模式
#     segFileData = jieba.cut(fileData, cut_all=True)
#
#     # 对空格，换行符进行处理
#     result = []
#     for data in segFileData:
#         data = ''.join(data.split())
#         if (data != '' and data != "\n" and data != "\n\n"):
#             result.append(data)
#
#     # 将分词后的结果用空格隔开，保存至本地。比如"我来到北京清华大学"，分词结果写入为："我 来到 北京 清华大学"
#     #f = open(sFilePath + "/" + filename + "-seg.txt", "w+")
#     recordFileName = segFileName.strip('.txt') + '_seg.txt'
#     recordFilePath = os.path.join(recordPath, recordFileName)
#     f = open(recordFilePath, "w+")
#     f.write(' '.join(result))
#     f.close()


# 读取100份已分词好的文档，进行TF-IDF计算
def tf_idf_method(file_list, count = 10, file_dir = 'training_data'):
    # 保留分词结果
    seg_data = {}
    keyword_data = {}
    # 存取100份文档的分词结果
    corpus = []
    for file_name in file_list:

        file = open(os.path.join(file_dir, file_name), 'r')
        content = file.read()
        file.close()
        corpus.append(content)
        seg_data[file_name] = content

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    vec = vectorizer.fit_transform(corpus)
    # tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    tfidf = transformer.fit_transform(vec)

    # then we save the model for our test file
    feature_path = 'models/tfidf_vec_vocabulary.pkl'
    tfidf_transformer_path = 'models/tfidf_transformer.pkl'

    if not os.path.exists('models'):
        os.mkdir('models')

    with open(feature_path, 'wb') as file:
        pickle.dump(vectorizer.vocabulary_, file)

    with open(tfidf_transformer_path, 'wb') as file:
        pickle.dump(transformer, file)

    '''
        then we also record the method to loading the model
        # 加载特征
        feature_path = 'models/feature.pkl'
        loaded_vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb")))
        # 加载TfidfTransformer
        tfidftransformer_path = 'models/tfidftransformer.pkl'
        tfidftransformer = pickle.load(open(tfidftransformer_path, "rb"))
        #测试用transform，表示测试数据，为list
        test_tfidf = tfidftransformer.transform(loaded_vec.transform(test_content))
        
    '''

    # ------------------------ 如果只进行模型保存的话， 不需要用以下内容 ------------ #

    # word = vectorizer.get_feature_names()  # 所有文本的关键字
    # weight = tfidf.toarray()  # 对应的tfidf矩阵\
    #
    # tfidf_save_path = 'tfidfFile'
    # if not os.path.exists(tfidf_save_path):
    #     os.mkdir(tfidf_save_path)

    # ------------------------------- 以下内容暂时不需要用 ------------------------------------------- #
    # 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
    # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    # for i in range(100):    # len(weight)
    #     print("--------Writing all the tf-idf in the", i, u" file into --------")
    #     tfidf_file_name = file_list[i] + '_tfidf.txt'
    #     # 暂不进行文件的写入，只进行关键值的记录
    #     # tfidfFilePath = os.path.join(tfidfPath, tfidfFileName)
    #     # f = open(tfidfFilePath, 'w+')
    #     word_data = {}
    #     for j in range(len(word)):
    #         # 进行排序输出
    #         word_data[word[j]] = weight[i][j]
    #         # f.write(word[j] + "    " + str(weight[i][j]) + "\n")
    #     sorted_word_data = sorted(word_data.items(), key=lambda d: d[1], reverse=True)  # 降序排列
    #     print(sorted_word_data[:10])
    #     print('-------------------------')
    #
    #     # 提取指定个数的关键词
    #     keyword_data_list = []
    #     # 进行文件存储
    #     flag = 0
    #     for line in sorted_word_data:
    #         if flag < count:
    #             keyword_data_list.append(line[0])
    #         # f.write(line[0])
    #         # f.write('\t')
    #         # f.write(str(line[1]))
    #         # f.write('\n')  # 显示写入换行
    #         flag += 1
    #     keyword_data[file_list[i]] = keyword_data_list
    #     # f.close()
    # return keyword_data, seg_data


if __name__ == "__main__":
    # (allfile, path) = getFilelist(sys.argv)
    # for ff in allfile:
    #     print "Using jieba on " + ff
    #     fenci(ff, path)
    # Tfidf(allfile)
    # fileList = getCorpusFilelist()
    # for file in fileList:
    #     print "Using jieba on " + file
    #     segFile(file)
    file_name_list = get_file_list('training_data')
    # keyword_data, seg_datas = tf_idf_method(file_name_list)

    tf_idf_method(file_name_list)

    # print(keyword_data)

