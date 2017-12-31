# -*- coding: utf-8 -*-
"""
Datasets
========

"""
from __future__ import absolute_import
from . import utils_datasets
import librosa
import numpy as np
import os


def load_jamendo(save_path='datasets', sr=16000, mono=True, duration=None, offset=0.0):
    """Download jamendo http://www.mathieuramona.com/wp/data/jamendo/
    It creates `save_path/jamendo` and the sub-directories, `jamendo_lab`, `train`, `valid`, `test`.

    It does not remove the downloaded `.tar.gz` files in `save_path/jamendo`.

    As it takes quite a while for decoding the audio files, it would be better to store
    the returned value as npy/hdf/whatever and use it.

    Parameters
    ----------
    save_path: str,
        absolute/relative path to store the dataset

    sr: int > 0
        sampling rate of audio sources.
        It is also used to compute label arrays

    mono: bool
        Whether downmix the audio signals to mono or not.

    duration: float [second]
        Duration of audio files

    offset: float [second]
        Offset to load

    Returns
    -------
        srcs: list, length of 3.
            |  each element is a list of train/valid/test sources
            |  e.g. ``srcs[0][0].shape = (1, 3959745)`` when ``sr=16000`` and ``mono=True``

        ys: list, length of 3.
            |  each element is a list of train/valid/test groundtruths
            |  e.g., ``ys[0][0].shape = (3959745, )``

    """
    set_names = ['train', 'valid', 'test']
    for set in set_names:
        datadir = utils_datasets.get_file('jam_{}_audio.tar.gz'.format(set),
                                          'http://www.mathieuramona.com/uploads/Main/jam_{}_audio.tar.gz'.format(set),
                                          save_path, untar=True, cache_subdir='jamendo',
                                          tar_folder_name=set)
    datadir = utils_datasets.get_file('jam_annote.tar.gz',
                                      'http://www.mathieuramona.com/uploads/Main/jam_annote.tar.gz',
                                      save_path, untar=True, cache_subdir='jamendo',
                                      tar_folder_name='jamendo_lab')
    # load file names in train/valid/test folder
    x_filenames = []
    y_filenames = []
    for set in set_names:
        fnames = [f.lstrip('._') for f in os.listdir(os.path.join(save_path, 'jamendo', set)) \
                  if f.split('.')[-1] in ('ogg', 'mp3')]
        x_filenames.append(fnames)
        y_filenames.append([f.split('.')[0] + '.lab' for f in fnames])

    srcs = []
    ys = []
    for set, x_fnames, y_fnames in zip(set_names, x_filenames, y_filenames):
        srcs_set = []
        ys_set = []
        for idx, (x_fname, y_fname) in enumerate(zip(x_fnames, y_fnames)):
            # process srcs
            print('Loading {}/{}: {}...'.format(idx, len(x_fnames), x_fname))
            src, _ = librosa.load(os.path.join(save_path, 'jamendo', set, x_fname),
                                  sr=sr, mono=mono, offset=offset, duration=duration)
            if mono:
                src = src[np.newaxis, :]  # to make it (1, N) instead of (N,)
            len_src = src.shape[1]
            srcs_set.append(src)
            # process ys
            y = np.zeros((len_src,), dtype=np.bool)
            with open(os.path.join(save_path, 'jamendo', 'jamendo_lab', y_fname)) as f_label:
                for line in f_label:
                    start, end, label = line.rstrip('\n').split(' ')
                    if label == 'sing':
                        start, end = int(np.round(float(start) * sr)), int(np.round(float(end) * sr))
                        y[start:end] = True
            ys_set.append(y)

        srcs.append(srcs_set)
        ys.append(ys_set)

    # return
    return srcs, ys


def load_fma(save_path='datasets', size='small'):
    """Download fma:free music archive (https://github.com/mdeff/fma)

    It would be better to download directly from the link for large/full..

    Parameters
    ----------
    save_path: str,
        absolute/relative path to store the dataset

    size: str, 'small', 'medium', 'large', 'huge'
        |  small: 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB)
        |  medium: 25,000 tracks of 30s, 16 unbalanced genres (22 GiB)
        |  large: 106,574 tracks of 30s, 161 unbalanced genres (93 GiB)
        |  full: 106,574 untrimmed tracks, 161 unbalanced genres (879 GiB)

    """
    assert size in ('small', 'medium', 'large', 'full')
    if size == 'small':
        zip_filename = 'fma_small.zip'
        zip_path = utils_datasets.get_file(zip_filename, 'https://os.unil.cloud.switch.ch/fma/fma_small.zip',
                                           save_path, untar=False, cache_subdir='fma',
                                           md5_hash='4edb51c99a19d31fe01a7d44d5cac19b')

    elif size == 'medium':
        zip_filename = 'fma_medium.zip'
        zip_path = utils_datasets.get_file(zip_filename, 'https://os.unil.cloud.switch.ch/fma/fma_medium.zip',
                                           save_path, untar=False, cache_subdir='fma')
    elif size == 'large':
        zip_filename = 'fma_large.zip'
        zip_path = utils_datasets.get_file(zip_filename, 'https://os.unil.cloud.switch.ch/fma/fma_large.zip',
                                           save_path, untar=False, cache_subdir='fma')
    elif size == 'full':
        zip_filename = 'fma_full.zip'
        zip_path = utils_datasets.get_file(zip_filename, 'https://os.unil.cloud.switch.ch/fma/fma_full.zip',
                                           save_path, untar=False, cache_subdir='fma')

    print("unzipping audio files...")
    os.system('unzip {} -d {}'.format(os.path.join(zip_path, zip_filename), zip_path))

    metadata_zip_filename = 'fma_metadata.zip'
    metadata_zip_path = utils_datasets.get_file(metadata_zip_filename,
                                                'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip',
                                                save_path, untar=False, cache_subdir='fma',
                                                md5_hash='d3ebfd86e283345ee2366a5492495935')
    print("unzipping metadata files...")
    os.system('unzip {} -d {}'.format(os.path.join(metadata_zip_path, metadata_zip_filename),
                                      metadata_zip_path))


def load_musicnet(save_path='datasets', format='hdf'):
    """Download musicnet (https://homes.cs.washington.edu/~thickstn/start.html)

    Parameters
    ----------
    save_path: str,
        absolute/relative path to store the dataset

    format: str,
        Data format to download. Either 'hdf' or 'npz'

    """
    assert format in ('hdf', 'npz')
    if format == 'hdf':
        utils_datasets.get_file('musicnet.h5', 'https://homes.cs.washington.edu/~thickstn/media/musicnet.h5',
                                save_path, untar=False, cache_subdir='musicnet',
                                md5_hash='05103753391a8019029b29b790f7e1f7')
    else:
        utils_datasets.get_file('musicnet.npz', 'https://homes.cs.washington.edu/~thickstn/media/musicnet.npz',
                                save_path, untar=False, cache_subdir='musicnet',
                                md5_hash='9303e5338adefd3715c51997553fb45f')
    utils_datasets.get_file('musicnet_metadata.csv',
                            'https://homes.cs.washington.edu/~thickstn/media/musicnet_metadata.csv',
                            save_path, untar=False, cache_subdir='musicnet',
                            md5_hash=None)


def load_magnatagatune(save_path='datasets'):
    """Download magnatagatune dataset, concate the zip files, unzip it,
    to `save_path`.

    Parameters
    ----------
        save_path: absolute or relative path to store the dataset

    """
    # 1GB for each
    zip_path = utils_datasets.get_file('mp3.zip.001', 'http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001',
                                       save_path, untar=False, cache_subdir='magnatagatune',
                                       md5_hash='179c91c8c2a6e9b3da3d4e69d306fd3b')
    utils_datasets.get_file('mp3.zip.002', 'http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002',
                            save_path, untar=False, cache_subdir='magnatagatune',
                            md5_hash='acf8265ff2e35c6ff22210e46457a824')
    utils_datasets.get_file('mp3.zip.003', 'http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003',
                            save_path, untar=False, cache_subdir='magnatagatune',
                            md5_hash='582dc649cabb8cd991f09e14b99349a5')

    print("appending zip files...")
    os.system('cat {}/mp3.zip.* > {}/mp3s.zip'.format(zip_path, zip_path))
    print("unzipping...")
    os.system('unzip {} -d {}/mp3s'.format(os.path.join(zip_path, 'mp3s.zip'), zip_path))
    # labels
    utils_datasets.get_file('clip_info_final.csv',
                            'http://mi.soi.city.ac.uk/datasets/magnatagatune/clip_info_final.csv',
                            save_path, untar=False, cache_subdir='magnatagatune',
                            md5_hash='03ef3cb8ddcfe53fdcdb8e0cda005be2')
    utils_datasets.get_file('annotations_final.csv',
                            'http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv',
                            save_path, untar=False, cache_subdir='magnatagatune',
                            md5_hash='f04fa01752a8cc64f6e1ca142a0fef1d')
    utils_datasets.get_file('comparisons_final.csv',
                            'http://mi.soi.city.ac.uk/datasets/magnatagatune/comparisons_final.csv',
                            save_path, untar=False, cache_subdir='magnatagatune')

    # echonest feature (377.4 MB)
    utils_datasets.get_file('mp3_echonest_xml.zip',
                            'http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3_echonest_xml.zip',
                            save_path, untar=False, cache_subdir='magnatagatune',
                            md5_hash='09be4ac8c682a8c182279276fadb37f9')


def load_gtzan_speechmusic(save_path='datasets'):
    """
    Download gtzan speech/music dataset, untar it, and create a helper csv file

    Arguments
    ---------
    save_path: str,
        Absolute or relative path to store the dataset

    """
    datadir = utils_datasets.get_file('gtzan_speechmusic.tar.gz', 'http://opihi.cs.uvic.ca/sound/music_speech.tar.gz',
                                      save_path, untar=True, cache_subdir='gtzan_speechmusic',
                                      md5_hash='b063639094c169062940becacd3108a0')

    labels = ['music', 'speech']

    rows = utils_datasets.get_rows_from_folders(folder_dataset='music_speech',
                                                folders=labels,
                                                dataroot=datadir)
    columns = ['id', 'filepath', 'label']
    csv_path = os.path.join(datadir, 'dataset_summary_kapre.csv')
    utils_datasets.write_to_csv(rows=rows, column_names=columns,
                                csv_fname=csv_path)


def load_gtzan_genre(save_path='datasets'):
    """Load gtzan muusic dataset from http://opihi.cs.uvic.ca/sound/genres.tar.gz
    It downloads gtzan tarball on save_path/gtzan_music.tar.gz .
    After untarring, we got files as below:
    ```
    for genre_name in ['blues', ..., 'rock']:
        for idx in xrange(100):
            "genres/{}/{}.{:05d}.au".format(genre_name, genre_name, idx)
    ```
    Then it creates a helper csv file.

    Parameters
    ----------
    save_path: str,
        Absolute or relative path to store the dataset

    """

    datadir = utils_datasets.get_file('gtzan_genre.tar.gz', 'http://opihi.cs.uvic.ca/sound/genres.tar.gz',
                                      save_path, untar=True, cache_subdir='gtzan_genre',
                                      md5_hash='fe37942310e589be16b04b6d631790de')
    labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    rows = utils_datasets.get_rows_from_folders(folder_dataset='genres',
                                                folders=labels,
                                                dataroot=datadir)
    columns = ['id', 'filepath', 'label']
    csv_path = os.path.join(datadir, 'dataset_summary_kapre.csv')
    utils_datasets.write_to_csv(rows=rows, column_names=columns,
                                csv_fname=csv_path)
