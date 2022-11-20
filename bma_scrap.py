from urllib.request import Request, urlopen
from selenium import webdriver
import time
import urllib


if __name__ == '__main__':
    driver = webdriver.Firefox()
    driver.get('http://www.bmatraffic.com/index.aspx')
    time.sleep(5)
    driver.get('http://www.bmatraffic.com/show.aspx?image=123')
    # u = 'http://www.bmatraffic.com/show.aspx?image=123&time=1668908014437'
    img = driver.find_element('xpath', '//img')
    with open('test.jpg', "wb") as f:
        f.write(img.screenshot_as_png)

    driver.close()
    # root_req = Request(u, headers={'User-Agent': 'Mozilla/5.0'})
    # text = urlopen(root_req).read()
    # with open('test.jpg', "wb") as f:
    #     f.write(text)

