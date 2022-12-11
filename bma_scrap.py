from urllib.request import Request, urlopen
from selenium import webdriver
import time
import urllib


if __name__ == '__main__':
    driver = webdriver.Firefox()
    driver.get('http://www.bmatraffic.com/index.aspx')
    time.sleep(20)
    while True:
        time.sleep(5)
        driver.get('http://www.bmatraffic.com/PlayVideo.aspx?ID=1606')
        img = driver.find_element('xpath', '//img')
        with open(f'/media/palm/Data/bma/1484/{time.time()}.jpg', "wb") as f:
            f.write(img.screenshot_as_png)
        time.sleep(120)

    driver.close()
    # root_req = Request(u, headers={'User-Agent': 'Mozilla/5.0'})
    # text = urlopen(root_req).read()
    # with open('test.jpg', "wb") as f:
    #     f.write(text)

