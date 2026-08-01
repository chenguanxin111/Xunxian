import urllib.request

for topic in ['/usb_cam/image_raw', '/xunxian/debug/overlay', '/xunxian/debug/hsv_mask']:
    url = 'http://127.0.0.1:8080/stream?topic=' + topic
    try:
        req = urllib.request.urlopen(url, timeout=3)
        print(topic, 'STATUS:', req.status, req.headers.get('Content-Type'))
    except Exception as e:
        print(topic, 'ERROR:', e)
