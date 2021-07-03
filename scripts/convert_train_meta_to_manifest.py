#!/usr/bin/env python3
import sys
import json
import pandas as pd
import soundfile as sf


def modify_target_text(target_str, mask='*'):
    """Insert a character that masks a word 'тысяч(-а/-и)'"""
    if len(target_str) > 3:
        target_list = list(target_str)
        target_list.insert(-3, mask)
        return ''.join(target_list)
    return target_str


def main(meta_path, save_path):
    # read meta
    meta = pd.read_csv(meta_path)
    meta.path = 'data/numbers/' + meta.path
    meta.number = meta.number.astype(str)
    meta['duration'] = meta['path'].apply(lambda x: sf.info(x).duration)

    # apply preprocessing to the text to deal with 'тысяча(-и)'
    meta.number = meta.number.apply(lambda x: modify_target_text(x))

    # write manifest
    with open(save_path, 'w') as fout:
        for idx, row in meta.iterrows():
            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": row['path'],
                "duration": row['duration'],
                "text": row['number']
            }
            json.dump(metadata, fout)
            fout.write('\n')
    print('Success!')


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python3 convert_meta_to_manifest.py <path_to_meta.csv> <save_path.json>')
    meta_path = sys.argv[1]
    save_path = sys.argv[2]
    main(meta_path, save_path)
