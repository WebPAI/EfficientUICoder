import os
from playwright.sync_api import sync_playwright
import argparse
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
MAX_PIXELS = 178956970

def take_screenshot(url, output_file="screenshot.png", do_it_again=False):
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    if os.path.exists(output_file) and not do_it_again:
        print(f"{output_file} exists!")
        return

    try:
        with sync_playwright() as p:
            # Choose a browser, e.g., Chromium, Firefox, or WebKit
            browser = p.chromium.launch()
            page = browser.new_page()

            # Navigate to the URL
            page.goto(url, timeout=60000)

            # Take the screenshot
            page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)

            browser.close()

        # --- 检查像素数，如果过大直接生成白图 ---
        try:
            # 暂时关闭 Pillow 的限制
            Image.MAX_IMAGE_PIXELS = None

            with Image.open(output_file) as img:
                num_pixels = img.width * img.height
                if num_pixels > MAX_PIXELS:
                    print(f"[Warning] Screenshot too large ({num_pixels} pixels), replacing with blank image. File: {os.path.abspath(output_file)}")
                    img = Image.new('RGB', (1280, 960), color='white')
                    img.save(output_file)
        except Exception as e_img:
            print(f"[Warning] Failed to open screenshot: {e_img}. Generating blank image.")
            img = Image.new('RGB', (1280, 960), color='white')
            img.save(output_file)

    except Exception as e: 
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image 
        img = Image.new('RGB', (1280, 960), color = 'white')
        img.save(output_file)


if __name__ == "__main__":

    # Initialize the parser
    parser = argparse.ArgumentParser(description='Process two path strings.')

    # Define the arguments
    parser.add_argument('--html', type=str)
    parser.add_argument('--png', type=str)

    # Parse the arguments
    args = parser.parse_args()

    take_screenshot(args.html, args.png, do_it_again=True)
