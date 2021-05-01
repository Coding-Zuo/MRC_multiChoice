# -*- coding:utf-8 -*-
import errno
import http.client
import requests
import random
import json
from hashlib import md5
import time

httpClient = None


class Trans(object):
    def __init__(self):
        self.appid = '20180603000171354'
        self.appkey = '1e_ozkWCRpvPTc5NNW5_'
        self.headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        self.salt = random.randint(32768, 65536)
        requests.adapters.DEFAULT_RETRIES = 5
        self.url = 'http://api.fanyi.baidu.com' + '/api/trans/vip/translate'

    def translate(self, content, from_lang, to_lang):
        sign = self.make_md5(self.appid + content + str(self.salt) + self.appkey)
        payload = {'appid': self.appid, 'q': content, 'from': from_lang, 'to': to_lang, 'salt': self.salt, 'sign': sign}
        requests.DEFAULT_RETRIES = 5  # 增加重试连接次数
        s = requests.session()
        s.keep_alive = False  # 关闭多余连接
        try:
            r = requests.post(self.url, params=payload, headers=self.headers, timeout=300)
            time.sleep(1)
        except Exception as e:
            if e.errno != errno.ECONNRESET:
                raise
            pass
        finally:
            if httpClient:
                httpClient.close()

        result = r.json()
        if "error_code" in result.keys():
            print("error!!!!:" + result['error_msg'])
            return "error!!!!:" + result['error_msg']

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
