import numpy as np
import matplotlib.pyplot as plt

def get_num_classes(labels):
    """
    retrives total number of classes.
    # Arguments:
        labels: list, label values:
    # Returns:
        int, total number of classes.
    # Raises:
        ValueError: if any label value in the range(0, num_classes -1)
            is missing or if number of classes is <=1.
    """
    # sparse categorical labels
    num_classes = max(labels) + 1
    # check for missing_classes
    missing_classes = [i for i in range(num_classes) if i not in labels]
    if len(missing_classes):
        raise ValueError('Missing samples with label value(s) '
                         '{missing_classes}. Please make sure you have '
                         'at least one sample for every label value '
                         'in the range(0, {max_class})'.format(
                            missing_classes=missing_classes,
                            max_class=num_classes - 1))

    if num_classes <= 1:
        raise ValueError('Invalid number of labels: {num_classes}.'
                         'Please make sure there are at least two classes '
                         'of samples'.format(num_classes=num_classes))
    return num_classes

def get_num_words_per_sample(sample_texts):
    """Gets the median number of words per sample given corpus.
    # Arguments
        sample_texts: list, sample texts.
    # Returns
        int, median number of words per sample.
    """
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)
