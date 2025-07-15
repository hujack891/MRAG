import requests
from bs4 import BeautifulSoup

# 设置请求头，防止被网站识别为爬虫
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
}

# 目标网页 URL
url = "https://www.ign.com/wikis/black-myth-wukong/Laurel_Buds"

# 发起 GET 请求
response = requests.get(url, headers=headers)
response.raise_for_status()  # 如果请求失败将抛出异常

# 使用 BeautifulSoup 解析 HTML
soup = BeautifulSoup(response.text, "html.parser")

# 提取所有 <img> 标签
img_elements = soup.find_all("img")

# 遍历并打印 src 地址
for img in img_elements:
    src = img.get("src") or img.get("data-src")  # 有些图片使用懒加载
    if src:
        print("✅ Image:", src)
