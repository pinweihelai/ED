import insuranceqa_data as insuranceqa
_train_data = insuranceqa.load_pairs_train()
_test_data = insuranceqa.load_pairs_test()
_valid_data = insuranceqa.load_pairs_valid()
vocab_data = insuranceqa.load_pairs_vocab()
print("keys", vocab_data.keys())
vocab_size = len(vocab_data['word2id'].keys())
