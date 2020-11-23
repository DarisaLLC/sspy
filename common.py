
from pathlib import Path
import datetime
import exifread
import time
import os


def numeric_sorted_files(dir_path, ext_wildcard='*.tif', post_fix='_segmented'):
    def isInt(value):
        try:
            int(value)
            return True
        except ValueError:
            return False

    file_paths = list(Path(dir_path).glob(ext_wildcard))
    pn = len(post_fix)

    for ff in file_paths:
        if pn > 0:
            numeric = ff.name.split('.')[0][-(6+pn):-pn]
        else:
            numeric = ff.name.split('.')[0][-6:]
        if not isInt(numeric):
            print(' %s Last 6 characters can not be converted to integer ' % numeric )
            return None
    if pn > 0:
        file_paths.sort(key=lambda f: int(f.name.split('.')[0][-(6+pn):-pn]))
    else:
        file_paths.sort(key=lambda f: int(f.name.split('.')[0][-6:]))

    return file_paths


def make_timed_dir_name(parent, prename):
    return os.path.join(parent, prename + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f'))


def file_modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def get_unix_time_from_filetime(date):
    return time.mktime(date.timetuple())


def get_unix_time_from_exif(exif_time, verbose):
    # Get EXIF Creation Date
    tokens = exif_time.split(' ')
    assert (len(tokens) >= 2)
    exif_date = tokens[0]
    exif_time = tokens[1]
    if verbose: print(exif_date, exif_time)
    (y, mm, d) = exif_date.split(':')
    (h, min, s) = exif_time.split(':')
    date_list = (y, mm, d, h, min, s)
    if verbose: print(date_list)
    #       int_date_list = [map(int, i) for i in date_list]
    # WTF&lt; why the H is that this:
    # (’2006′, ’18′, ’12′, ’17′, ’18′, ’26′)
    # [[2, 0, 0, 6], [1, 8], [1, 2], [1, 7], [1, 8], [2, 6]]
    int_date_list = map(int, date_list)
    if verbose: print("int_date_list:", int_date_list)
    (y, mm, d, h, min, s) = int_date_list
    if verbose: print(y, mm, d, h, min, s)
    # create unix timestamp object from EXIF date
    exif_dateTaken = datetime.datetime (y, mm, d, h, min, s)
    if verbose: print(str(exif_dateTaken))
    tdict = {}
    tdict['exif_date'] = exif_date
    tdict['exif_time'] = exif_time
    tdict['parts'] = (y, mm, d, h, min, s)
    tdict['unixtime'] = time.mktime(exif_dateTaken.timetuple())
    return tdict


def get_timestamp_from_exif(filepath):
    with open(filepath, 'rb') as img_file:
        try:
            tags = exifread.process_file(img_file, stop_tag="DateTimeOriginal", details=False)
        except:
            return None  # if a file is corrupt in any way, return None
        date = tags.get("EXIF DateTimeOriginal") or tags.get("EXIF DateTimeDigitized") \
               or tags.get("Image DateTime")

        return get_unix_time_from_exif(date.printable, False)


