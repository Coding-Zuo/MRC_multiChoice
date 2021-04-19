# -*- coding:utf-8 -*-


import requests
import random
import json
from hashlib import md5
import time


class Trans(object):
    def __init__(self):
        self.appid = '20210419000788725'
        self.appkey = '2WeK52AEBcyrqnvAEVCp'
        self.headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        self.salt = random.randint(32768, 65536)
        self.url = 'http://api.fanyi.baidu.com' + '/api/trans/vip/translate'

    def translate(self, content, from_lang, to_lang):
        sign = self.make_md5(self.appid + content + str(self.salt) + self.appkey)
        payload = {'appid': self.appid, 'q': content, 'from': from_lang, 'to': to_lang, 'salt': self.salt, 'sign': sign}
        r = requests.post(self.url, params=payload, headers=self.headers)
        time.sleep(1)
        result = r.json()
        result = result['trans_result'][0]['dst']
        return result

    def make_md5(self, s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()


if __name__ == '__main__':
    appid = '20210419000788725'
    appkey = '2WeK52AEBcyrqnvAEVCp'
    # For list of language codes, please refer to `https://api.fanyi.baidu.com/doc/21`
    from_lang = 'en'
    to_lang = 'zh'

    endpoint = 'http://api.fanyi.baidu.com'
    path = '/api/trans/vip/translate'
    url = endpoint + path

    query = 'Hello World! This is 1st paragraph.\nThis is 2nd paragraph.'


    # Generate salt and sign
    def make_md5(s, encoding='utf-8'):
        return md5(s.encode(encoding)).hexdigest()


    salt = random.randint(32768, 65536)
    sign = make_md5(appid + query + str(salt) + appkey)

    # Build request
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}

    # Send request
    r = requests.post(url, params=payload, headers=headers)
    result = r.json()

    # Show response
    print(json.dumps(result, indent=4, ensure_ascii=False))
