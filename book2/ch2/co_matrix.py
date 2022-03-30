import sys
sys.path.append('../..')
import numpy as np
from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
print(id_to_word)

def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_if = corpus[right_idx]
                co_matrix[id, right_word_if] += 1
    
    return co_matrix

C = create_co_matrix(corpus, len(word_to_id), 1)
print(C)