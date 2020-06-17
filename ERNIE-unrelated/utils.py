import os
import sys
import torch
import pickle
import pandas as pd

from torch.utils.data import TensorDataset
from tqdm import tqdm


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_eval_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the eval set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the text set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class ATSAProcessor(DataProcessor):
    """Processor for the ATSA data set (NLPCC 2020)."""
    
    def __init__(self):
        self.df_train = pd.read_pickle('./data/train.pkl')
        self.df_eval = pd.read_pickle('./data/dev.pkl')
        self.df_test = pd.read_pickle('./data/test.pkl') # ---------- change to test file at last!  ----------

    def get_train_examples(self, args):
        return self._create_examples(self.df_train, labels_available=True, do_lower=args.do_lower_case)

    def get_eval_examples(self, args):
        return self._create_examples(self.df_eval, labels_available=True, do_lower=args.do_lower_case)

    def get_test_examples(self, args):
        return self._create_examples(self.df_test, labels_available=False, do_lower=args.do_lower_case)

    def get_labels(self):
        """See base class."""
        return [0, 1, 2]

    @staticmethod
    def _create_examples(df, labels_available=True, do_lower=True):
        """Creates examples for the training/eval and test sets."""
        examples = []
        for i in range(len(df)):
            guid = i
            text_a = df.text[i]
            text_b = df.term[i] # text_b=None if ignore aspect information
            # text_b = df.format_a[i] 

            if do_lower == True:
                text_a = text_a.lower()
                if text_b:
                    text_b = text_b.lower()

            if labels_available:
                labels = df.label[i]
            else:
                labels = -1

            examples.append(
                InputExample(
                        guid=guid, 
                        text_a=text_a,
                        text_b=text_b,
                        labels=labels)
                        )
        return examples


def convert_examples_to_features(args, examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (example_index, example) in tqdm(enumerate(examples), desc="Convert examples"):
        # if example_index % 1000 == 0:
        #     print("Converting examples %d of %d" % (ex_index, len(examples)))
        
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        token_type_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            token_type_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        token_type_ids += padding
        
        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        # if example_index < 5:
        #     print("*** Example ***")
        #     print("guid: %s" % (example.guid))
        #     print("tokens: %s" % " ".join([str(x) for x in tokens]))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

        features.append(
                InputFeatures(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            label_id=float(example.labels)))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_Dataset(args, processor, tokenizer, mode="train"):
    if mode == "train":
        examples = processor.get_train_examples(args)
    elif mode == "eval":
        examples = processor.get_eval_examples(args)
    elif mode == "test":
        examples = processor.get_test_examples(args)
    else:
        raise ValueError("mode must be one of train, eval, or test")

    features = convert_examples_to_features(args, examples, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)

    return examples, features, data
