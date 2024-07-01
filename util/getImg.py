# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:02:46 2020

@author: MSI-NB

从百度爬取图片做测试
"""

import requests
import re
from urllib.parse import quote


def get_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0"
    }
    req = requests.get(url=url, headers=headers)
    return req.text


def get_jpg(source):
    data_list = re.findall(r'"thumbURL":"(.*?)",', source)
    return data_list


def get_bytes(url_one):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:79.0) Gecko/20100101 Firefox/79.0"
    }
    rep = requests.get(url=url_one, headers=headers)
    return rep.content


# 图片保存路径
def save_pic(content, numb):
    with open("D:/Chorm_download/img/images/train/{}.jpg".format(numb), 'ab+') as fp:
        fp.write(content)


def main():

    name = input("请输入搜索图片的关键词: ")
    key_word = quote(name)
    page_num = input("请输入下载图片的数量: ")
    req = re.findall(r'(\d+)', page_num)

    while len(req) == 0:
        page_num = input("输入格式不对！请重新输入下载图片的数量: ")
        req = re.findall(r'(\d+)', page_num)
    num = int(req[0])
    if num % 30 == 0:
        page = num//30
    else:
        page = num//30+1
    print(num)
    for ele in range(30, page*30+30, 30):
        url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj\
        &ct=201326592&is=&fp=result&queryWord={}&cl=2&lm=-1&ie=utf-8&oe=utf-8&\
        adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&word={}&s=&se=&tab=&width=&height=&\
        face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn={}&rn=30=".format(key_word, key_word, ele)
        html = get_text(url)  # 返回每一个网址的源代码
        jpg_list = get_jpg(html)  # 返回每一个网址内的图片地址
        i = 1  # 每一张图的序号
        for jpg in jpg_list:
            if int(ele)-30+i == num+1:
                break
            print('第%d张图片正在下载' % (int(ele)-30+i))
            content = get_bytes(jpg)  # 将每一张图片源代码的字节形式返回
            save_pic(content, ele-30+i)  # 将每一张图保存到本地
            print('第%d图片下载完成' % (int(ele)-30+i))
            i += 1  # 随着每一张图的保存，序号相应加一，30张图保存后,i再次赋值为1代表每一个网址的第一张图


main()
