# -*- coding: utf-8 -*-

import logging
import tarfile
from urllib.request import urlretrieve
import os
import re
import collections
import numpy as np
import random

"""
Global setting
"""
LOG_FORMAT = '%(asctime)s - [%(levelname)s]-[%(name)s]: %(message)s'
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger('WORD2VEC USEING TF2.0 with IMDBb')
logger.setLevel(logging.INFO)

# Training Parameters.
learning_rate = 0.1
global_batch_size = 128
num_steps = 3000000
display_step = 10000
eval_step = 200000

# Word2Vec Parameters.
# Dimension of the embedding vector.
embedding_size = 200
# Total number of different words in the vocabulary.
max_vocabulary_size = 50000
# Remove all words that does not appears at least n times.
min_occurrence = 10
# How many words to consider left and right.
global_skip_window = 3
# How many times to reuse an input to generate a label.
global_num_skips = 2
# Number of negative examples to sample
num_sampled = 64
# Generate training batch index
data_index = 0

# 下載
url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
file_path = os.curdir + os.sep + 'data' + os.sep + 'aclImdb_v1.tar.gz'
data_dir = os.curdir + os.sep + 'data' + os.sep
data_Path = os.curdir + os.sep + 'data' + os.sep + 'aclImdb'
if not os.path.isfile(file_path):
    logger.info('Downloading from {}'.format(url))
    result = urlretrieve(url, file_path)
    logger.info('Download: {}'.format(result))
else:
    logger.info('Download was success')

# 解壓
if not os.path.isdir(data_Path):
    logger.info('Extracting {} to data'.format(file_path))
    tar_file = tarfile.open(file_path, 'r:gz')
    result = tar_file.extractall(data_dir)
    logger.info('Extracting {} to data over'.format(file_path))
else:
    logger.info('Extracting data was success'.format(file_path))


def rm_tags(text):
    """
    r: 正則表示式和 \ 會有衝突，
    當一個字串使用了正則表示式後，最好在前面加上'r'
    :param text:
    :return:
    """
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_files(file_type):
    """
    讀取imdb資料

    :param file_type:   "train" or "test"
    :return:    Tuple(List of labels, List of articles)
    """
    file_list = []
    positive_path = os.path.join(os.path.join(data_Path, file_type), 'pos')
    for f in os.listdir(positive_path):
        file_list.append(os.path.join(positive_path, f))

    negative_path = os.path.join(os.path.join(data_Path, file_type), 'neg')
    for f in os.listdir(negative_path):
        file_list.append(os.path.join(negative_path, f))

    logger.info('Read {} with {} files'.format(file_type, len(file_list)))

    # 創建label，後續有擴充程式時，用得到
    all_labels = ([1] * 12500 + [0] * 12500)
    all_texts = []
    for fi in file_list:
        # logger.info('Read {}'.format(fi))
        with open(fi, encoding='utf8') as fh:
            all_texts += [rm_tags(' '.join(fh.readlines()))]

    return all_labels, all_texts


train_labels, train_text = read_files('train')
logger.info('train_text type {}、size: {}'.format(type(train_text), len(train_text)))
# logger.info('train_text 第一筆資料: {}'.format(train_text[0]))

# 將所有文章內容文字轉換為list，for後續word2vec處理
train_word = []
for item in train_text:
    train_word.extend([x.replace('.', '') for x in item.split()])

logger.info('train_word size: {}'.format(len(train_word)))

# Build the dictionary and replace rare words with UNK token.(unknown word)
count = [('UNK', -1)]
count.extend(collections.Counter(train_word).most_common(max_vocabulary_size - 1))

# Remove samples with less than 'min_occurrence' occurrences.
for i in range(len(count) - 1, -1, -1):
    if count[i][1] < min_occurrence:
        count.pop(i)
    # The collection is ordered, so stop when 'min_occurrence' is reached.
    else:
        break

vocabulary_size = len(count)
logger.info('vocabulary size: {}'.format(vocabulary_size))

# Assign an id to each word
word2id = dict()
for i, (word, _) in enumerate(count):
    word2id[word] = i

# logger.info('word2id size: {}'.format(len(word2id)))

# 如果出現未知的文字，放入unk；
data = list()
unk_count = 0
for word in train_word:
    index = word2id.get(word, 0)
    if index == 0:
        unk_count += 1
    data.append(index)

count[0] = ('UNK', unk_count)

# Assign a word to each an id
id2word = dict(zip(word2id.values(), word2id.keys()))
# logger.info('id2word size: {}'.format(len(id2word)))

logger.info('Words count: {}'.format(len(train_word)))
logger.info('Unique words: {}'.format(len(set(train_word))))
logger.info('Vocabulary size: {}'.format(vocabulary_size))
logger.info('Most common words: {}'.format(count[:10]))


# Generate training batch for the skip-gram model.
def next_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # get window size (words left and right + current one).
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for idx in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[idx * num_skips + j] = buffer[skip_window]
            labels[idx * num_skips + j, 0] = buffer[context_word]

        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch.
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

