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
from selenium.common.exceptions import NoSuchElementException
import subprocess
import contextlib

# Add paths and configuration
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging
from config import BaseConfig

baseconfig = BaseConfig()
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])
logger.info("Start executing IGN Wiki scraping program")

# === Initialize file paths and configuration ===
out_folder = "./data"
file_name = 'ign_wiki_links_all.csv'
os.makedirs(out_folder, exist_ok=True)
csv_file_path = os.path.join(out_folder, file_name)

# === Initialize WebDriver ===
chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--log-level=3')  



service = Service(executable_path=baseconfig.CHROME_DRIVER_PATH,log_path=os.devnull)  
service.creationflags = subprocess.CREATE_NO_WINDOW  

# Suppress stderr before starting ChromeDriver
with open(os.devnull, 'w') as fnull, contextlib.redirect_stderr(fnull):
    driver = webdriver.Chrome(service=service, options=chrome_options)

logger.info("Browser initialized successfully")

# === Incremental update of links ===
wiki_hrefs = set()
if os.path.exists(csv_file_path):
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)  
        next(reader, None)  # Skip the header row
        for row in reader:
            if row:
                wiki_hrefs.add(row[0])
    logger.info(f"The number of loaded historical links: {len(wiki_hrefs)}")
else:
    # If the file does not exist, create it and write the header
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Link'])
    logger.info("Create a new CSV file and write the header information.")

# === open the website ===
driver.get('https://www.ign.com/wikis')
time.sleep(2 + random.uniform(0, 1))

# Handling the privacy policy pop-up window
try:
    consent_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I Consent')]"))
    )
    consent_button.click()
    logger.info("Clicked on the privacy policy pop-up window")
except Exception:
    logger.warning("No privacy policy pop-up was detected or the click failed.")

# Scroll the page to the bottom of the web page.
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

# Set the sorting order to be in alphabetical order from A to Z.
select_element = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, 'select[aria-label="Sort"]'))
)
select = Select(select_element)
select.select_by_value('AlphaAZ')
time.sleep(2 + random.uniform(0, 1))

# === Start iteration and click "Load" ===
all_save_num = 0
loop_num = 0
# A -> Z
try:
    while True:
        # After clicking "load more" 100 times, read the link once.
        
        for i in range(100):
            logger.info(f"Round {i} - Clicking Load more")
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(1,2))
            try:
                load_more_button = driver.find_element(By.XPATH, '//a[contains(text(), "Load More")]')
                load_more_button.click()
                time.sleep(random.uniform(2, 3))
            except NoSuchElementException:
                logger.warning("No 'Load More' button found, exiting while loop.")
                raise StopIteration  # ⬅️ 通过异常退出外层 while

        save_link_num = 0    
        tile_links = driver.find_elements(By.CSS_SELECTOR, 'a.tile-link')  
        logger.info("Save the link")
        for link in tile_links:
            href = link.get_attribute('href')
            if href and '/wikis/' in href and href not in wiki_hrefs:
                wiki_hrefs.add(href)
                save_link_num += 1
                with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([href])
        logger.info(f'Total links saved in this batch: {save_link_num}')
        all_save_num += save_link_num
        logger.info(f'{all_save_num} links have been saved so far')
        logger.info(f'Looped {loop_num} times so far')
        loop_num += 1

except StopIteration:
    logger.info("Finished: No more 'Load More' button.")
except Exception as e:
    logger.error(f"Unexpected error: {e}")

# Save the last batch of links
save_link_num = 0  
loop_num += 1
tile_links = driver.find_elements(By.CSS_SELECTOR, 'a.tile-link')  
logger.info("Save the last batch of links")
for link in tile_links:
    href = link.get_attribute('href')
    if href and '/wikis/' in href and href not in wiki_hrefs:
        wiki_hrefs.add(href)
        save_link_num += 1
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([href])
all_save_num += save_link_num
logger.info(f'{all_save_num} links have been saved so far')
logger.info(f'Looped {loop_num} times so far')
logger.info('All links have been saved')


