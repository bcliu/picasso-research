import os
import sys
import urllib
import json
import multiprocessing
from skimage import io

SORT_BY_OPTIONS = [ 'interestingness-desc', 'relevance', 'date-taken-desc' ]
DOWNLOAD_URL_KEY = 'url_l'

API_KEY = 'dac1f5e9f956e8cc64a3bf1d1141c86c'
SORT_BY = SORT_BY_OPTIONS[1]

query = sys.argv[1]
query_ = query.replace(' ', '_')
path = sys.argv[2]
num_to_download = int(sys.argv[3])
start_from_page = int(sys.argv[4])

print 'Searching for ' + query + ', sorted by ' + SORT_BY
print 'Downloading ' + str(num_to_download)

num_downloaded = 0
current_page = start_from_page - 1

file_urls = []
file_paths = []

while num_downloaded <= num_to_download:
    current_page += 1
    params = urllib.urlencode({
        'method': 'flickr.photos.search',
        'api_key': API_KEY,
        'text': query,
        'safe_search': 1,
        'content_type': 1,
        'media': 'photos',
        'per_page': 300,
        'format': 'json',
        'nojsoncallback': 1,
        'sort': SORT_BY,
        'extras': DOWNLOAD_URL_KEY,  # Retrieve download URL
        'page': current_page
    })

    url = 'https://api.flickr.com/services/rest/' + '?' + params
    decoded = json.loads(urllib.urlopen(url).read().decode('utf-8'))
    photos = decoded['photos']['photo']

    for photo in photos:
        if DOWNLOAD_URL_KEY in photo:
            url = photo.get(DOWNLOAD_URL_KEY)
            file_urls.append(url)
            file_paths.append(os.path.join(path, query_ + str(num_downloaded) + '.jpg'))
            num_downloaded += 1
            sys.stdout.write("\rFinished: %d%%" % num_downloaded)
            sys.stdout.flush()

            if num_downloaded >= num_to_download:
                print 'Done retrieving all image URLs'
                break


def download_image(args_tuple):
    """For use with multiprocessing map. Returns filename on fail."""
    try:
        url, filename = args_tuple
        if not os.path.exists(filename):
            urllib.urlretrieve(url, filename)
        test_read_image = io.imread(filename)
        return True
    except KeyboardInterrupt:
        raise Exception()  # multiprocessing doesn't catch keyboard exceptions
    except:
        return False


print 'Downloading...'
num_workers = 16
if num_workers <= 0:
    num_workers = multiprocessing.cpu_count() + num_workers
pool = multiprocessing.Pool(processes=num_workers)
map_args = zip(file_urls, file_paths)
results = pool.map(download_image, map_args)
