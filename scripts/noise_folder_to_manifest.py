#!/usr/bin/env python3
import sys
import json

import glob
import numpy as np
import soundfile as sf


def main(noise_path, label, save_path, segment_duration, max_len=3.83):
    # find all wavs
    filenames = sorted(list(glob.glob(f'{noise_path}/**/*.wav', recursive=True)))
    durations = [sf.info(filename).duration for filename in filenames]

    # write manifest
    with open(save_path, 'w') as fout:
        for filepath, duration in zip(filenames, durations):

            # skipping file if it is very short
            if duration < segment_duration:
                continue

            # Generate random file start (in seconds)
            max_offset = duration - segment_duration
            if max_offset == 0:
                offset = 0
            else:
                offset = np.random.randint(0, max_offset)

            # Write the metadata to the manifest
            metadata = {
                'audio_filepath': filepath,
                'duration': segment_duration,
                'label': label,
                'text': '_',  # for compatibility with ASRAudioText collection
                'offset': offset
            }
            json.dump(metadata, fout)
            fout.write('\n')

    print('Success!')


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python3 noise_folder_to_manifest.py <noises_path> <label> <save_path.json> <segment_duration>=1.')
        sys.exit()

    noise_path = sys.argv[1]
    label = sys.argv[2]
    save_path = sys.argv[3]

    segment_duration = 1.
    if len(sys.argv) > 4:
        segment_duration = float(sys.argv[4])

    main(noise_path, label, save_path, segment_duration)
