import urllib.request

for url in [
    'http://127.0.0.1:5001/api/status',
    'http://127.0.0.1:8080/stream?topic=/right_turn/debug/overlay',
    'http://127.0.0.1:8080/stream?topic=/right_turn/debug/mask'
]:
    try:
        req = urllib.request.urlopen(url, timeout=3)
        print(url, 'STATUS:', req.status, req.headers.get('Content-Type'))
    except Exception as e:
        print(url, 'ERROR:', e)
