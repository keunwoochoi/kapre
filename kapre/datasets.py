import os
import utils_datasets


def load_magnatagatune(save_path='datasets'):
    """Download magnatagatune dataset, concate the zip files, unzip it,
    to `save_path`.

    Arguments
    ---------
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
        save_path: absolute or relative path to store the dataset

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

    Arguments
    ---------
        save_path: absolute or relative path to store the dataset

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


if __name__ == "__main__":
    print("Test")
    load_gtzan_genre()
    load_magnatagatune()
    load_gtzan_speechmusic()
    print('done')
