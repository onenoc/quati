import torchtext

from quati.dataset.fields.words import WordsField
from quati.dataset.fields.tags import TagsField
from quati.dataset.corpora.corpus import Corpus


class SSTCorpus(Corpus):
    task = 'doc'

    """
    Args:
        fields_tuples (list): passed to the constructor.
        lazy (bool): passed to the constructor.
        subtrees (bool): whether to include sentiment-tagged subphrases
            in addition to complete examples. Default: False
        granularity (str): indicate the classification polarity type:
            2: negatives vs positives (neutrals are ignored);
            3: negatives, neutrals and positives;
            5: very neg, neg, neutrals, pos, very pos.
            Default: 2
        return_invalid_targets (bool): use examples without a label.
            Default: False
    """
    def __init__(
        self, fields_tuples, lazy=False, subtrees=False, granularity='2',
        return_invalid_targets=False,
    ):
        super().__init__(fields_tuples, lazy=lazy)
        self.subtrees = subtrees
        self.granularity = granularity
        self.return_invalid_targets = return_invalid_targets
        self.granularity_map = {
            '0': 'very negative',
            '1': 'negative',
            '2': 'neutral',
            '3': 'positive',
            '4': 'very positive',
            None: None
        }
        if granularity == '2':
            self.granularity_map['0'] = 'negative'
            self.granularity_map['2'] = None
            self.granularity_map['4'] = 'positive'
        elif granularity == '3':
            self.granularity_map['0'] = 'negative'
            self.granularity_map['4'] = 'positive'

    @staticmethod
    def create_fields_tuples():
        fields_tuples = [
            ('words', WordsField()),
            ('target', TagsField())
        ]
        return fields_tuples

    def _read(self, file):
        for line in file:
            examples = self.make_torchtext_example(line.strip())
            for ex in examples:
                if hasattr(ex, 'target') and ex.target is None:
                    if not self.return_invalid_targets:
                        continue
                yield ex

    def make_torchtext_example(self, text, label=None):
        from nltk.tree import Tree
        tree = Tree.fromstring(text)
        if self.subtrees:
            examples = []
            for subtree in tree.subtrees():
                ex = {'words': ' '.join(subtree.leaves()),
                      'target': self._get_label(subtree.label())}
                if 'target' not in self.fields_dict.keys():
                    del ex['target']
                assert ex.keys() == self.fields_dict.keys()
                examples.append(
                    torchtext.data.Example.fromdict(ex, self.fields_dict)
                )
            return examples
        else:
            ex = {'words': ' '.join(tree.leaves()),
                  'target': self._get_label(tree.label())}
            if 'target' not in self.fields_dict.keys():
                del ex['target']
            assert ex.keys() == self.fields_dict.keys()
            return [torchtext.data.Example.fromdict(ex, self.fields_dict)]

    def _get_label(self, label):
        return self.granularity_map[label]


if __name__ == '__main__':
    from quati.dataset.corpora.test_corpus import quick_test
    quick_test(
        SSTCorpus,
        '../../../data/corpus/sst/train.txt',
        lazy=True,
        subtrees=False,
        granularity='2'
    )
