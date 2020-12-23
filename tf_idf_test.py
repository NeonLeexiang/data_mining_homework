"""
    date:       2020/12/23 3:42 下午
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
import pickle  # we use pickle to save our model file


def load_model(vocabulary_path, transformer_path):
    """
    this function is to load the model object and return
    :param vocabulary_path: path of pickle file vocabulary
    :param transformer_path: path of pickle file transformer
    :return: vocabulary, transformer
    """
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
    # loading feature
    loaded_vec = CountVectorizer(decode_error='replace', vocabulary=pickle.load(open(vocabulary_path, 'rb')))
    # loading tfidf transformer
    tfidf_transformer = pickle.load(open(transformer_path,'rb'))
    return loaded_vec, tfidf_transformer


def tf_idf_test(txt_src_path, txt_trg_path, type='F1'):
    test_vectorizer, test_tfidf_transformer = load_model('models/tfidf_vec_vocabulary.pkl',
                                                         'models/tfidf_transformer.pkl')
    # ------------------------------- trg_dict -----------------------
    seg_dict = {}
    keyword_dict = {}

    corpus = []

    with open(txt_src_path, 'r') as file:
        lines = file.readlines()
        line_count = 1
        for line in lines:
            corpus.append(line)
            name = str(line_count).zfill(6)
            seg_dict[name] = line
            line_count += 1

    # ------------------------------- trg_dict -----------------------
    trg_dict = {}

    with open(txt_trg_path, 'r') as file:
        lines = file.readlines()
        line_count = 1
        for line in lines:
            name = str(line_count).zfill(6)
            trg_dict[name] = line
            line_count += 1



# 读取100份已分词好的文档，进行TF-IDF计算
def tf_idf_method(training_file,count=10, tfidf_file_path=None):
    # 保留分词结果
    # seg_data = {}
    # keyword_data = {}
    # 存取100份文档的分词结果
    corpus = []

    with open(training_file, 'r') as file:
        lines = file.readlines()
        line_count = 1
        for line in lines:
            corpus.append(line)
            # name = str(line_count).zfill(6)
            # seg_data[name] = line
            line_count += 1

    # for file_name in file_list:
    #     file = open(os.path.join(file_dir, file_name), 'r')
    #     content = file.read()
    #     file.close()
    #     corpus.append(content)
    #     seg_data[file_name] = content

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
    # file_name_list = get_file_list('training_data')
    # keyword_data, seg_datas = tf_idf_method(file_name_list)

    file_path = 'datasets/train_src.txt'

    tf_idf_method(file_path)

    # print(keyword_data)
