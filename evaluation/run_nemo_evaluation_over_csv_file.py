#!/usr/bin/env python3
import sys
import random

import nemo.collections.asr as nemo_asr
import pandas as pd


def postprocess_text(text: str) -> str:
    """Add additional text post-processing.
    For instance, clean '*' symbol.
    Also if such symbol detected, make sure that at
    least 3 digits follow this symbol (or sample 
    randomly at least =) )

    Args:
        text (str): Decoded string

    Returns:
        str: Postprocessed string
    """
    if '*' in text:
        # finds first occurence of '*', if there are many
        asterisk_id = text.index('*')
        # remove all '*'
        text = text.replace('*', '')
        # get left and right splits
        left_part, right_part = text[: asterisk_id], text[asterisk_id: ]

        # check how many digits contains right part of a number
        if len(right_part) == 2:
            # if 2 digits are given, randomly copy any of them,
            # as both equally plausible 
            chance = random.random()
            if chance >= 0.5:
                # copy right digit
                right_part = right_part + right_part[1]
            else:
                # copy left digit
                right_part = right_part[0] + right_part
        elif len(right_part) == 1:
            # copy digit 3 times
            right_part = ''.join([right_part] * 3)
        elif len(right_part) == 0:
            # randomly generate 3 digits
            right_part = list()
            for i in range(3):
                right_part.append(str(random.randint(0, 9)))
            right_part = ''.join(right_part)

        return ''.join([left_part, right_part])
    else:
        return text


def test():
    a = ['332*4', '332*44', '332*46', '332*', '332*456', '3324']
    for item in a:
        print(postprocess_text(item))


def split_dataframe(df, chunk_size=10000):
    num_chunks = df.shape[0] // chunk_size
    for i in range(num_chunks):
        yield df[i * chunk_size: (i + 1) * chunk_size]


def main(artefact_path, csv_path, save_path, batch_size):

    # read csv file
    df = pd.read_csv(csv_path)

    # load model
    model = nemo_asr.models.EncDecCTCModel.restore_from(
        artefact_path
    )

    # run inference
    results = list()
    for chunk in split_dataframe(df, batch_size):
        batch_size = chunk.shape[0]
        result = model.transcribe(
            paths2audio_files=chunk['path'].to_list(),
            batch_size=batch_size
        )
        results.extend(result)

    df['number'] = [postprocess_text(item) for item in results]
    df.to_csv(save_path, index=False)
    print(f'Success, check {save_path}')


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python3 run_nemo_evaluation_over_csv_file.py artefacts/QuartzNet5x1Aug.nemo',
              '<test_csv_path>.csv <result_csv_path>.csv <batch_size>=32')
        sys.exit()

    batch_size = 32
    if len(sys.argv) > 4:
        batch_size = int(sys.argv[4])

    artefact_path = sys.argv[1]
    csv_path = sys.argv[2]
    save_path = sys.argv[3]

    main(artefact_path, csv_path, save_path, batch_size)
