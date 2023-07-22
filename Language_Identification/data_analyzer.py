# ============================ Third Party libs ============================
import os
import logging
import numpy as np
import pandas as pd
import math
import nagisa
import jieba
from matplotlib import pyplot as plt
# ============================ My packages ============================
from Language_Identification.data_loader import read_csv
from Language_Identification.configurations import BaseConfig
from Language_Identification.utils import LABEL2LANGUAGE

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    CONFIG_CLASS = BaseConfig()
    ARGS = CONFIG_CLASS.get_config()

    TRAIN_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.train_data_file))
    VALID_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.valid_data_file))
    TEST_DATA = read_csv(os.path.join(ARGS.raw_data_dir, ARGS.test_data_file))

    logging.info("Train set contain %s sample ...", len(TRAIN_DATA))
    logging.info("Dev set contain %s sample ...", len(VALID_DATA))
    logging.info("Test set contain %s sample ...\n", len(TEST_DATA))

    NUM_LABELS = len(np.unique(TRAIN_DATA.labels.values))
    logging.info("The %s different classes are existed in dataset.\n", NUM_LABELS)

    GROUPED_DATA_FRAME = TRAIN_DATA.groupby("labels")
    for name, group in GROUPED_DATA_FRAME:
        SAMPLE_LENGTHS = []
        for sample in group.text.values:
            if name == "zh":
                tokenized = jieba.lcut(sample)
            elif name == "ja":
                tokenized = nagisa.tagging(sample).words
            else:
                tokenized = sample.split()
            SAMPLE_LENGTHS.append(len(tokenized))

        logging.info("Sample description of %s", name)
        logging.info(pd.DataFrame({"len": SAMPLE_LENGTHS}).describe())
        logging.info("\n")
        SAMPLE_LENGTHS_BINS = np.linspace(math.ceil(min(SAMPLE_LENGTHS)),
                                          math.floor(max(SAMPLE_LENGTHS)),
                                          50)
        plt.hist(SAMPLE_LENGTHS, bins=SAMPLE_LENGTHS_BINS, alpha=0.5)
        plt.title(
            f"Histogram for {LABEL2LANGUAGE[name]}")  # Set the title for the current histogram
        plt.xlabel("Sample Length")
        plt.ylabel("Frequency")
        plt.xlim(0, 100)
        plt.ylim(0, 1800)

        plt.savefig(
            os.path.join(ARGS.plots_dir,
                         ARGS.sample_length_analysis_folder) + f"/{LABEL2LANGUAGE[name]}.png")
        plt.clf()
        # plt.show()
