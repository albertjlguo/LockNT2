#!/usr/bin/env python3
"""
YouTube Cookies Support Script
用于生产环境绕过YouTube机器人验证的cookies支持工具
"""

import os
import json
import logging
from pathlib import Path

def create_cookies_file():
    """创建cookies.txt文件模板"""
    cookies_template = """# Netscape HTTP Cookie File
# This is a generated file! Do not edit.

# YouTube cookies for bypassing bot detection
# Format: domain	flag	path	secure	expiration	name	value

# Example cookies (replace with your actual cookies):
# .youtube.com	TRUE	/	TRUE	1234567890	VISITOR_INFO1_LIVE	your_visitor_info_here
# .youtube.com	TRUE	/	TRUE	1234567890	YSC	your_ysc_here
# .youtube.com	TRUE	/	TRUE	1234567890	PREF	your_pref_here

# To get your cookies:
# 1. Open YouTube in browser
# 2. Login if needed
# 3. Use browser dev tools > Application > Cookies
# 4. Copy the cookie values above
"""
    
    cookies_path = Path("cookies.txt")
    if not cookies_path.exists():
        with open(cookies_path, 'w', encoding='utf-8') as f:
            f.write(cookies_template)
        print(f"Created cookies template at: {cookies_path.absolute()}")
        return str(cookies_path.absolute())
    else:
        print(f"Cookies file already exists at: {cookies_path.absolute()}")
        return str(cookies_path.absolute())

def validate_cookies_file(cookies_path):
    """验证cookies文件格式"""
    if not os.path.exists(cookies_path):
        return False, "Cookies file not found"
    
    try:
        with open(cookies_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查是否包含实际的cookie数据（非注释行）
        lines = [line.strip() for line in content.split('\n') 
                if line.strip() and not line.strip().startswith('#')]
        
        if not lines:
            return False, "No actual cookie data found (only comments)"
        
        # 检查基本格式
        for line in lines:
            parts = line.split('\t')
            if len(parts) < 7:
                return False, f"Invalid cookie format in line: {line}"
        
        return True, f"Found {len(lines)} valid cookies"
        
    except Exception as e:
        return False, f"Error reading cookies file: {str(e)}"

def get_cookies_instructions():
    """获取cookies设置说明"""
    instructions = """
🍪 YouTube Cookies 设置说明

生产环境下YouTube可能会要求登录验证，需要提供浏览器cookies来绕过限制。

📋 获取Cookies步骤：

1. 打开浏览器，访问 https://youtube.com
2. 如果需要，完成登录和验证
3. 打开开发者工具 (F12)
4. 切换到 Application/应用程序 标签
5. 左侧选择 Storage > Cookies > https://www.youtube.com
6. 复制以下重要cookies的值：
   - VISITOR_INFO1_LIVE
   - YSC  
   - PREF
   - CONSENT (如果有)

📝 填写cookies.txt文件：

将cookies按以下格式填入cookies.txt：
.youtube.com	TRUE	/	TRUE	1234567890	VISITOR_INFO1_LIVE	你的值
.youtube.com	TRUE	/	TRUE	1234567890	YSC	你的值
.youtube.com	TRUE	/	TRUE	1234567890	PREF	你的值

⚠️  注意事项：
- 使用TAB分隔符，不是空格
- 过期时间可以设置为未来的时间戳
- 保护好cookies文件，不要提交到版本控制

✅ 完成后重启应用即可使用cookies绕过验证
"""
    return instructions

if __name__ == "__main__":
    print("🔧 YouTube Cookies Support Setup")
    print("=" * 50)
    
    # 创建cookies文件
    cookies_path = create_cookies_file()
    
    # 验证现有cookies
    is_valid, message = validate_cookies_file(cookies_path)
    print(f"\n📋 Cookies validation: {message}")
    
    if not is_valid:
        print("\n" + get_cookies_instructions())
        print(f"\n📁 Please edit the file: {cookies_path}")
    else:
        print("✅ Cookies file is ready for use!")
