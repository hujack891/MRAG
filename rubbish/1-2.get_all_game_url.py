import sys, os
import csv
import time
import random
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import subprocess
import contextlib

# 添加路径和配置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging
from config import CHROME_DRIVER_PATH

logger = setup_logging("1.get_all_game_url")
logger.info("开始执行 IGN Wiki 抓取程序")

# === 初始化文件路径与配置 ===
out_folder = "./data"
os.makedirs(out_folder, exist_ok=True)
csv_file_path = os.path.join(out_folder, 'ign_wiki_links_all.csv')

# === 初始化 WebDriver ===
chrome_options = Options()
# chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--log-level=3')  # 设置日志等级

service = Service(executable_path=CHROME_DRIVER_PATH,log_path=os.devnull)  # 屏蔽 chromedriver.log 文件)
service.creationflags = subprocess.CREATE_NO_WINDOW  # Windows下避免弹窗

# 启动前屏蔽 stderr
with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
    driver = webdriver.Chrome(service=service, options=chrome_options)

logger.info("浏览器初始化成功")

# === 读取已有链接 ===
wiki_hrefs = set()
if os.path.exists(csv_file_path):
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)  # 把这个文件变成一个可以逐行读取的迭代器，每一行是一个列表。
        next(reader, None)  # 跳过表头
        for row in reader:
            if row:
                wiki_hrefs.add(row[0])
    logger.info(f"已加载历史链接数量: {len(wiki_hrefs)}")
else:
    # 如果文件不存在，创建并写入表头
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Link'])
    logger.info("创建新CSV文件并写入表头")

# === 打开目标页面 ===
driver.get('https://www.ign.com/wikis')
time.sleep(2 + random.uniform(0, 1))

# 处理隐私弹窗
try:
    consent_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I Consent')]"))
    )
    consent_button.click()
    logger.info("已点击隐私弹窗")
except Exception:
    logger.warning("未检测到隐私弹窗或点击失败")

# 页面滚动
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

# 设置排序为按字母顺序
select_element = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, 'select[aria-label="Sort"]'))
)
select = Select(select_element)
select.select_by_value('AlphaZA')
time.sleep(2 + random.uniform(0, 1))

# === 开始迭代点击加载 ===
load_more_clicks = 0
max_clicks = 10000  # 最大轮次
try:
    while load_more_clicks <= max_clicks:
        logger.info(f"第 {load_more_clicks + 1} 轮 - 正在提取链接")
        # 读取页面上的全部链接
        tile_links = driver.find_elements(By.CSS_SELECTOR, 'a.tile-link')
        new_count = 0
        # 读取还没有保存的链接写到csv文件中
        for link in tile_links:
            href = link.get_attribute('href')
            if href and '/wikis/' in href and href not in wiki_hrefs:
                wiki_hrefs.add(href)
                new_count += 1
                with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([href])

                # 检测是不是最后一款游戏
                if '/wikis/007-legends' in href:
                    logger.warning("检测到目标链接 007-legends，提前终止爬取")
                    raise StopIteration

        logger.info(f"本轮新增链接: {new_count}，总计: {len(wiki_hrefs)}")

        try:
            # 点击Load More
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            load_more_button = driver.find_element(By.XPATH, '//a[contains(text(), "Load More")]')
            load_more_button.click()
            time.sleep(0.5 + random.uniform(0, 1))
        except Exception as e:
            logger.warning("未能找到或点击 'Load More' 按钮，可能已到底部")
            break

        load_more_clicks += 1

except StopIteration:
    logger.info("爬取提前结束，已找到 007-legends")

finally:
    driver.quit()
    logger.info(f"所有链接已保存至：{csv_file_path}")
