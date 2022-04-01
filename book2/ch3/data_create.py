import sys
sys.path.append('..')
from common.util import convert_one_hot, preprocess, create_contexts_target

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_ward = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)
print(target)
print(contexts)