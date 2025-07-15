import os
import time
import sys
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By  # Missing import - this was causing the error
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from urllib.parse import urlparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logs.log_config import setup_logging
from urllib.parse import urlparse
logger = setup_logging(os.path.splitext(os.path.basename(__file__))[0])

def init_driver():
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    chrome_driver_path = os.getenv('CHROME_DRIVER_PATH')
    if chrome_driver_path:
        service = Service(chrome_driver_path)
    else:
        service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def get_active_slide(driver):
    """
    返回当前激活的侧边栏 slide，通过查找含有 "sidebar-content-slide" 和 "show" 类的容器。
    优先找包含返回按钮（Back）的区域。
    """
    try:
        # Try to find slide with back button first
        slide = driver.find_element(
            By.XPATH, 
            "//div[contains(@class, 'sidebar-content-slide') and contains(@class, 'show') and .//button[@data-cy='title-bar']]"
        )
        return slide
    except:
        try:
            # Fallback to any active slide
            return driver.find_element(By.CSS_SELECTOR, "div.sidebar-content-slide.show")
        except:
            # Last resort - find any visible sidebar content
            return driver.find_element(By.CSS_SELECTOR, "div.sidebar-content-slide")

def get_current_slide_title(driver):
    """获取当前激活slide的标题，用于状态验证"""
    try:
        active_slide = get_active_slide(driver)
        # 尝试多种方式获取标题
        title_selectors = [
            "h1", "h2", "h3", 
            ".title", ".heading", 
            "[data-cy='title-bar']",
            ".navigation-item.active"
        ]
        
        for selector in title_selectors:
            try:
                title_element = active_slide.find_element(By.CSS_SELECTOR, selector)
                title = title_element.text.strip()
                if title:
                    return title
            except:
                continue
        
        # 如果都没找到，返回slide的部分innerHTML作为标识
        soup = BeautifulSoup(active_slide.get_attribute("innerHTML"), "html.parser")
        text_content = soup.get_text()[:100]  # 取前100个字符作为标识
        return text_content.strip()
    except:
        return "unknown"

def normalize_text(text):
    """归一化文本：合并多个空白字符、转换为小写。"""
    return " ".join(text.split()).lower()

def wait_for_slide_change(driver, old_title, timeout=10):
    """等待slide变化"""
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: get_current_slide_title(d) != old_title
        )
        return True
    except TimeoutException:
        return False

def debug_page_structure(driver):
    """调试页面结构，找出所有可能的返回按钮"""
    try:
        print("=== 调试页面结构 ===")
        
        # 获取所有按钮
        buttons = driver.find_elements(By.TAG_NAME, "button")
        print(f"找到 {len(buttons)} 个按钮:")
        
        for i, btn in enumerate(buttons):
            try:
                text = btn.text.strip()
                class_name = btn.get_attribute("class")
                data_cy = btn.get_attribute("data-cy")
                title = btn.get_attribute("title")
                is_displayed = btn.is_displayed()
                is_enabled = btn.is_enabled()
                
                print(f"  按钮 {i+1}: 文本='{text}', 类='{class_name}', data-cy='{data_cy}', title='{title}', 可见={is_displayed}, 可用={is_enabled}")
                
                # 特别关注可能的返回按钮
                if any(keyword in str(attr).lower() for attr in [text, class_name, data_cy, title] 
                       for keyword in ['back', 'return', 'title-bar', '返回'] if attr):
                    print(f"    *** 可能的返回按钮 ***")
                    
            except Exception as e:
                print(f"  按钮 {i+1}: 获取信息失败 - {e}")
        
        # 获取当前激活的slide
        try:
            active_slide = get_active_slide(driver)
            slide_html = active_slide.get_attribute("outerHTML")[:500]  # 前500字符
            print(f"当前激活slide HTML片段: {slide_html}")
        except Exception as e:
            print(f"获取激活slide失败: {e}")
            
        print("=== 调试结束 ===")
        
    except Exception as e:
        print(f"调试页面结构失败: {e}")

def click_back_button(driver, max_retries=3):
    """点击返回按钮，带重试机制和多种定位策略"""
    # 根据你的截图，返回按钮在蓝框中，我们需要更精确的定位策略
    back_button_selectors = [
        # 基于你的截图，返回按钮应该在这些位置
        "//button[contains(@class, 'back') or contains(@class, 'return') or @data-cy='title-bar']",
        "//div[contains(@class, 'sidebar-content-slide') and contains(@class, 'show')]//button[1]",
        "//div[contains(@class, 'title-bar') or contains(@class, 'header')]//button",
        "//button[contains(@title, 'Back') or contains(@title, 'Return')]",
        "//button[contains(text(), 'Back') or contains(text(), '返回')]",
        "//div[contains(@class, 'navigation')]//button[1]",
        "//div[contains(@class, 'slide-header')]//button",
        # 更广泛的搜索
        "//button[@data-cy='title-bar']",
        "//button[contains(@class, 'back')]",
        "//button[contains(@class, 'return')]",
        "//a[contains(@class, 'back')]",
    ]
    
    for attempt in range(max_retries):
        print(f"尝试点击返回按钮 (尝试 {attempt + 1}/{max_retries})")
        
        # 首先调试页面结构
        if attempt == 0:
            debug_page_structure(driver)
        
        # 尝试多种定位策略
        for i, selector in enumerate(back_button_selectors):
            try:
                print(f"  使用选择器 {i+1}: {selector}")
                back_buttons = driver.find_elements(By.XPATH, selector)
                
                if not back_buttons:
                    print(f"    未找到按钮")
                    continue
                
                for j, back_btn in enumerate(back_buttons):
                    try:
                        print(f"    尝试按钮 {j+1}")
                        
                        # 检查按钮是否可见和可点击
                        if not back_btn.is_displayed():
                            print(f"      按钮不可见，跳过")
                            continue
                            
                        if not back_btn.is_enabled():
                            print(f"      按钮不可用，跳过")
                            continue
                        
                        # 滚动到按钮位置
                        driver.execute_script("arguments[0].scrollIntoView({behavior: 'instant'});", back_btn)
                        time.sleep(0.5)
                        
                        # 记录点击前的状态
                        before_click = get_current_slide_title(driver)
                        print(f"      点击前状态: {before_click}")
                        
                        # 多种点击方式
                        click_methods = [
                            lambda btn: btn.click(),  # 普通点击
                            lambda btn: driver.execute_script("arguments[0].click();", btn),  # JS点击
                            lambda btn: driver.execute_script("arguments[0].dispatchEvent(new Event('click'));", btn),  # 事件派发
                        ]
                        
                        for k, click_method in enumerate(click_methods):
                            try:
                                print(f"      尝试点击方式 {k+1}")
                                click_method(back_btn)
                                time.sleep(2)  # 等待页面响应
                                
                                # 检查点击是否生效
                                after_click = get_current_slide_title(driver)
                                print(f"      点击后状态: {after_click}")
                                
                                if before_click != after_click:
                                    print(f"    返回按钮点击成功！")
                                    return True
                                else:
                                    print(f"    点击后页面未变化")
                                    
                            except Exception as click_e:
                                print(f"      点击方式 {k+1} 失败: {click_e}")
                                continue
                                
                    except Exception as btn_e:
                        print(f"    按钮 {j+1} 处理失败: {btn_e}")
                        continue
                        
            except Exception as e:
                print(f"    选择器 {i+1} 失败: {e}")
                continue
        
        # 如果所有方法都失败，尝试按ESC键或其他方式
        try:
            print("  尝试按ESC键")
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
            time.sleep(1)
            
            # 检查ESC是否有效
            current_title = get_current_slide_title(driver)
            print(f"  ESC后当前标题: {current_title}")
            
        except Exception as esc_e:
            print(f"  ESC键失败: {esc_e}")
        
        if attempt < max_retries - 1:
            time.sleep(2)
    
    print("所有返回按钮点击方法都失败")
    return False

def verify_navigation_state(driver, expected_level, navigation_stack):
    """
    验证当前导航状态是否正确
    - expected_level: 期望的层级
    - navigation_stack: 导航栈，记录每层的标题
    """
    current_title = get_current_slide_title(driver)
    
    # 检查当前标题是否与期望层级匹配
    if expected_level < len(navigation_stack):
        expected_title = navigation_stack[expected_level]
        if normalize_text(current_title) == normalize_text(expected_title):
            return True
    
    return False

def recover_navigation_state(driver, target_level, navigation_stack, max_recovery_attempts=5):
    """
    恢复到目标导航层级
    - target_level: 目标层级
    - navigation_stack: 导航栈
    """
    print(f"尝试恢复导航状态到层级 {target_level}")
    
    for attempt in range(max_recovery_attempts):
        current_title = get_current_slide_title(driver)
        print(f"  当前标题: {current_title}")
        
        # 检查当前是否已经在正确层级
        if target_level < len(navigation_stack):
            expected_title = navigation_stack[target_level]
            print(f"  期望标题: {expected_title}")
            if normalize_text(current_title) == normalize_text(expected_title):
                print(f"成功恢复到层级 {target_level}")
                return True
        
        # 如果已经在根层级，无法再返回
        if target_level == 0:
            # 尝试刷新页面或其他方式回到根层级
            try:
                print("  尝试刷新页面回到根层级")
                current_url = driver.current_url
                driver.get(current_url)
                time.sleep(3)
                
                # 检查是否回到根层级
                root_title = get_current_slide_title(driver)
                if len(navigation_stack) > 0:
                    expected_root = navigation_stack[0]
                    if normalize_text(root_title) == normalize_text(expected_root):
                        print("通过刷新页面成功回到根层级")
                        return True
                
            except Exception as refresh_e:
                print(f"  刷新页面失败: {refresh_e}")
        
        # 尝试点击返回按钮
        print(f"  尝试点击返回按钮 (恢复尝试 {attempt + 1})")
        if not click_back_button(driver, max_retries=2):
            print(f"恢复失败：无法点击返回按钮 (尝试 {attempt + 1})")
            
            # 尝试其他恢复方式
            try:
                print("  尝试其他恢复方式...")
                # 点击页面空白处
                driver.execute_script("document.body.click();")
                time.sleep(1)
                
                # 尝试按多次ESC
                for _ in range(3):
                    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                    time.sleep(0.5)
                    
            except Exception as other_e:
                print(f"  其他恢复方式失败: {other_e}")
            
            time.sleep(2)
            continue
        
        time.sleep(2)
    
    print(f"无法恢复到层级 {target_level}")
    return False

def extract_active_slide(driver, level=0, visited_buttons=None, navigation_stack=None):
    """
    在当前激活的 slide 内提取所有导航项（按钮或链接）。
    增加了状态验证和恢复机制。
    """
    if visited_buttons is None:
        visited_buttons = set()
    if navigation_stack is None:
        navigation_stack = []
    
    output = []
    time.sleep(1)  # 等待当前 slide 更新

    try:
        active_slide = get_active_slide(driver)
    except Exception as e:
        print("No active slide found:", e)
        return output

    # 记录当前层级的标题到导航栈
    current_title = get_current_slide_title(driver)
    if level >= len(navigation_stack):
        navigation_stack.append(current_title)
    else:
        navigation_stack[level] = current_title

    soup = BeautifulSoup(active_slide.get_attribute("innerHTML"), "html.parser")
    # 找到当前 slide 内所有 navigation-item（按钮和链接）
    nav_items = soup.find_all(lambda tag: tag.name in ["button", "a"] 
                              and tag.has_attr("class") and "navigation-item" in tag["class"])
    
    for item in nav_items:
        label = item.get_text(strip=True)
        if not label:
            continue

        # 对于链接，直接输出；这里不做去重
        if item.name == "a":
            href = item.get("href")
            if href:
                full_url = f"https://www.ign.com{href}" if href.startswith("/") else href
                output.append(f"{'    '*level}- [{label}]({full_url})")
            continue

        # 对于按钮，使用 visited_buttons 防止重复点击
        if item.name == "button":
            if label in visited_buttons:
                continue
            visited_buttons.add(label)
            output.append(f"{'    '*level}- {label}")
            
            # 记录点击前的状态
            pre_click_title = get_current_slide_title(driver)
            
            try:
                # 在当前激活 slide 内查找所有按钮
                buttons = active_slide.find_elements(By.CSS_SELECTOR, "button.navigation-item")
                target_btn = None
                norm_label = normalize_text(label)
                for btn in buttons:
                    btn_text = normalize_text(btn.text)
                    if norm_label == btn_text or norm_label in btn_text:
                        target_btn = btn
                        break
                
                # 如果在当前 slide 未找到，则尝试全局搜索（利用 title 属性）
                if target_btn is None:
                    try:
                        target_btn = driver.find_element(By.XPATH, f"//button[@title='{label}']")
                    except Exception as e:
                        print(f"按钮 '{label}' 未在当前 slide 中定位到。")
                        continue

                driver.execute_script("arguments[0].scrollIntoView({behavior: 'instant'});", target_btn)
                target_btn.click()
                time.sleep(1.5)  # 等待子 slide 加载

                # 验证点击后的状态
                post_click_title = get_current_slide_title(driver)
                if normalize_text(post_click_title) == normalize_text(pre_click_title):
                    print(f"警告：点击按钮 '{label}' 后页面似乎没有变化")
                    # 可以在这里添加额外的重试逻辑
                    time.sleep(2)  # 额外等待
                    post_click_title = get_current_slide_title(driver)

                # 递归处理子 slide
                child_output = extract_active_slide(driver, level+1, visited_buttons, navigation_stack)
                output += child_output

                # 点击返回按钮回到父级 slide
                current_title_before_return = get_current_slide_title(driver)
                print(f"准备返回，当前标题: {current_title_before_return}")
                
                if not click_back_button(driver):
                    print(f"返回按钮点击失败，尝试恢复导航状态")
                    if not recover_navigation_state(driver, level, navigation_stack):
                        print(f"无法恢复到正确的导航层级，跳过后续处理")
                        break
                else:
                    # 验证是否成功返回到正确层级
                    if not wait_for_slide_change(driver, current_title_before_return, timeout=5):
                        print("返回操作可能失败，页面未发生变化")
                    
                    time.sleep(1)
                    current_title_after_return = get_current_slide_title(driver)
                    print(f"返回后标题: {current_title_after_return}")
                    
                    if not verify_navigation_state(driver, level, navigation_stack):
                        print(f"返回后状态验证失败，尝试恢复")
                        if not recover_navigation_state(driver, level, navigation_stack):
                            print(f"无法恢复到正确的导航层级，跳过后续处理")
                            break

            except Exception as e:
                print(f"处理按钮 '{label}' 时出错：", e)
                # 发生错误时也尝试恢复状态
                print("尝试恢复导航状态...")
                recover_navigation_state(driver, level, navigation_stack)
    
    return output

def get_sidebar_markdown(url):
    driver = init_driver()
    try:
        driver.get(url)

        try:
            consent_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'I Consent')]"))
            )
            consent_button.click()
            logger.info("已关闭隐私政策弹窗")
            time.sleep(3)
        except TimeoutException:
            logger.error("未找到隐私政策弹窗")
        time.sleep(2)

        time.sleep(5)  # 初始加载等待
        md_text = extract_active_slide(driver)
        return "\n".join(md_text)
    finally:
        driver.quit()

def save_markdown(md_text, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    wiki_url = "https://www.ign.com/wikis/minecraft"
    path = urlparse(wiki_url).path
    slug = path.strip('/').split('/')[-1]



    md_output = get_sidebar_markdown(wiki_url)
    print(md_output)
    output_path = f'data/{slug}_ign_sidebar.md'
    save_markdown(md_output,output_path)