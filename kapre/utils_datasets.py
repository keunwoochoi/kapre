"""data downloader. Keunwoo Choi, 14 Mar 2017.
Copied from keras.utils.data_utils and modify path/api a little.
99.999% credit to Keras.
"""
from __future__ import absolute_import
from __future__ import print_function

import functools
import tarfile
import os
import sys
import shutil
# import hashlib
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError
from six.moves.urllib.error import HTTPError

from keras.utils.generic_utils import Progbar
from keras.utils.data_utils import validate_file

import pandas as pd

allowed_exts = set(['mp3', 'wav', 'au', 'm4a'])

if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        """Replacement for `urlretrive` for Python 2.
        Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
        `urllib` module, known to have issues with proxy management.
        # Arguments
            url: url to retrieve.
            filename: where to store the retrieved data locally.
            reporthook: a hook function that will be called once
                on establishment of the network connection and once
                after each block read thereafter.
                The hook will be passed three arguments;
                a count of blocks transferred so far,
                a block size in bytes, and the total size of the file.
            data: `data` argument passed to `urlopen`.
        """

        def chunk_read(response, chunk_size=8192, reporthook=None):
            total_size = response.info().get('Content-Length').strip()
            total_size = int(total_size)
            count = 0
            while 1:
                chunk = response.read(chunk_size)
                count += 1
                if not chunk:
                    reporthook(count, total_size, total_size)
                    break
                if reporthook:
                    reporthook(count, chunk_size, total_size)
                yield chunk

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


def get_file(fname, origin, save_path, untar=False,
             md5_hash=None, cache_subdir='datasets', tar_folder_name=None):
    """Downloads a file from a URL if it not already in the cache.
    Passing the MD5 hash will verify the file after download
    as well as if it is already present in the cache.

    Usually it downloads the file to
        save_path/cache_dubdir/fname

    Arguments
    ---------
        fname: name of the file
        origin: original URL of the file
        save_path: path to create cache_subdir.
        untar: boolean, whether the file should be decompressed
        md5_hash: MD5 hash of the file for verification
        cache_subdir: directory being used as the cache
        tar_folder_name: string, if inside of abc.tar.gz is not abc but def, pass def here.

    Returns
    -------
        Path to the downloaded file
    """
    datadir_base = save_path
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.expanduser(os.path.join('~', '.kapre'))
        print('Given path {} is not accessible. Trying to use~/.kapre instead..')
        if not os.access(datadir_base, os.W_OK):
            print('~/.kapre is not accessible, using /tmp/kapre instead.')
            datadir_base = os.path.join('/tmp', '.kapre')
    datadir = os.path.join(datadir_base, cache_subdir)

    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        assert fname.endswith('.tar.gz'), fname
        fpath = os.path.join(datadir, fname)
        if tar_folder_name:
            untar_fpath = os.path.join(datadir, tar_folder_name)
        else:
            untar_fpath = fpath.rstrip('.tar.gz')
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if md5_hash is not None:
            if not validate_file(fpath, md5_hash):
                print('A local file was found, just checked md5 hash, but it might be '
                      'incomplete or outdated')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)
        progbar = None

        def dl_progress(count, block_size, total_size, progbar=None):
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath,
                            functools.partial(dl_progress, progbar=progbar))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            print('Untaring file...')
            tfile = tarfile.open(fpath, 'r:gz')
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
            # return untar_fpath

    return datadir


def write_to_csv(rows, column_names, csv_fname):
    '''write a csv file using given rows, columns, and filename

    Arguments
    ---------
        rows: list of rows (= which are lists.)
        column_names: names for columns
        csv_fname: string, csv file name

    Returns
    -------
        None
    '''
    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(csv_fname)


def get_rows_from_folders(folder_dataset, folders, dataroot=''):
    '''Utility function to create a csv file for a dataset.
    For gtzan music, gtzan speech, ballroom extended, jamendo,
    where each class in different folders.

    Arguments
    ---------
        folder_dataset: folder name in the dataset
        folders: name of subfolders. Usually same as class names.
        dataroot: base path of the dataset. Default is '' in case if you just wanna
            specify the whole path in folder_dataset.

    Returns
    -------
        list of lists, [file_id, file_path, file_label] where
            file_id: just file names
            file_path: file paths of each file (without data_root for more universal usage)
            file_label: integer, 0 to N-1, according to the order in `folders`.

    Example
    -------
    folders = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    rows = get_rows_from_folders('gtzan_music', folders, '/usr/keunwoo/datasets')
    '''
    rows = []
    for label_idx, folder in enumerate(folders):  # assumes different labels per folders.
        files = os.listdir(os.path.join(dataroot, folder_dataset, folder))
        files = [f for f in files if f.split('.')[-1].lower() in allowed_exts]
        for fname in files:
            file_path = os.path.join(folder_dataset, folder, fname)
            file_id = os.path.splitext(fname)[0]
            file_label = label_idx
            rows.append([file_id, file_path, file_label])
    return rows
