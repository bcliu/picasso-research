import os
import sys
import urllib
import json

SORT_BY_OPTIONS = [ 'interestingness-desc', 'relevance', 'date-taken-desc' ]
DOWNLOAD_URL_KEY = 'url_l'

API_KEY = 'dac1f5e9f956e8cc64a3bf1d1141c86c'
SORT_BY = SORT_BY_OPTIONS[1]

query = sys.argv[1]
query_ = query.replace(' ', '_')
path = sys.argv[2]
num_to_download = int(sys.argv[3])
start_from_page = int(sys.argv[4])
start_from_image_id = int(sys.argv[5])

print 'Searching for ' + query + ', sorted by ' + SORT_BY
print 'Downloading ' + str(num_to_download)

num_downloaded = start_from_image_id - 1
current_page = start_from_page - 1

while num_downloaded <= num_to_download:
    current_page = current_page + 1
    params = urllib.urlencode({
        'method': 'flickr.photos.search',
        'api_key': API_KEY,
        'text': query,
        'safe_search': 1,
        'content_type': 1,
        'media': 'photos',
        'per_page': 100,
        'format': 'json',
        'nojsoncallback': 1,
        'sort': SORT_BY,
        'extras': DOWNLOAD_URL_KEY, # Retrieve download URL
        'page': current_page
    })

    url = 'https://api.flickr.com/services/rest/' + '?' + params
    decoded = json.loads(urllib.urlopen(url).read().decode('utf-8'))
    photos = decoded['photos']['photo']

    for photo in photos:
        if DOWNLOAD_URL_KEY in photo:
            url = photo.get(DOWNLOAD_URL_KEY)
            image_path = path + '/' + query_ + '_' + str(num_downloaded) + '.jpg'
            print(url + '\t=>\t' + image_path)
            urllib.urlretrieve(url, image_path)
            num_downloaded = num_downloaded + 1

            if num_downloaded > num_to_download:
                print 'Done.'
                sys.exit(0)

print 'Done.'
