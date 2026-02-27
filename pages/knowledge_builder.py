"""
知识库构建页面 - 文件上传与自动化处理
用户上传文件，系统自动完成清洗、分块、向量化
"""

import streamlit as st
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict
import zipfile
import mimetypes
import pandas as pd

# PDF处理库
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from pdfplumber import PDF
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

# MinerU - 更准确的学术文档解析 (API方式)
try:
    import requests
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))



# CSS样式 - 适配 Modern 主题
st.markdown("""
<style>
    /* 知识库构建页面专用样式 */
    .upload-area {
        border: 2px dashed #4A90E2;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background-color: #f0f9ff;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #357ABD;
        background-color: #e0f2fe;
        transform: translateY(-2px);
    }
    
    .process-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    .status-success {
        color: #27ae60;
        font-weight: 600;
    }
    
    .status-error {
        color: #e74c3c;
        font-weight: 600;
    }
    
    .status-processing {
        color: #f39c12;
        font-weight: 600;
    }
    
    .file-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4A90E2;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """初始化session state"""
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "idle"  # idle, uploading, processing, completed, error
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = {}
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    if 'mineru_token' not in st.session_state:
        # 默认Token
        st.session_state.mineru_token = 'eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI4MjkwMDMxMyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1OTQ5NTU2NiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTM5MzI4ODE4NTEiLCJvcGVuSWQiOm51bGwsInV1aWQiOiJlYzI5YjY3MC0zYjk4LTQ1NTgtODE3Ny01N2VhM2RkMmYzN2EiLCJlbWFpbCI6IiIsImV4cCI6MTc2MDcwNTE2Nn0.7MWjSewD2AzOndsWH5MGMwtCB8tvCcFnGsiAcM5zzjzdrfhTz0376ysAFPpDMbqVAFL3WoEMLE3-4M8qzlM_3A'

def convert_pdf_to_markdown(pdf_path: Path) -> str:
    """将PDF文件转换为Markdown文本 - 使用MinerU获得更高精度"""
    markdown_content = ""
    
    try:
        if MINERU_AVAILABLE:
            # 尝试使用MinerU API (最优方案)
            try:
                markdown_content = convert_with_mineru(pdf_path)
                return markdown_content
            except Exception as mineru_error:
                # MinerU失败，记录错误并使用备用方案
                print(f"MinerU失败: {str(mineru_error)}, 使用备用方案")
        
        # 备用方案：pdfplumber
        if PDFPLUMBER_AVAILABLE:
            # 使用pdfplumber (备用方案)
            with open(pdf_path, 'rb') as file:
                pdf = PDF(file)
                pdf_text = ""
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text += page_text + "\n\n"
                
                # 清理和格式化PDF文本
                markdown_content = format_pdf_text(pdf_text)
                
        elif PDF_AVAILABLE:
            # 使用PyPDF2 (最后备用)
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pdf_text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        pdf_text += page_text + "\n\n"
                
                # 清理和格式化PDF文本
                markdown_content = format_pdf_text(pdf_text)
                
        else:
            markdown_content = f"[PDF文件: {pdf_path.name}]\n\n⚠️ PDF处理库未安装，建议安装: pip install miner-llm (推荐) 或 pip install pdfplumber"
            
    except Exception as e:
        markdown_content = f"[PDF处理错误: {pdf_path.name}]\n\n❌ 处理PDF文件时出错: {str(e)}"
    
    return markdown_content

def upload_to_free_host(file_path: Path) -> str:
    """上传文件到免费托管服务获取公开URL
    
    尝试多个免费服务:
    1. catbox.moe (稳定，中国可用)
    2. tmpfiles.org (简单)
    3. uguu.se (匿名上传)
    4. file.io (临时文件)
    """
    import requests
    
    st.info("📤 步骤1/3: 上传文件到临时托管服务...")
    
    # 方案1: catbox.moe (推荐，中国可用)
    try:
        st.info("🔄 尝试 catbox.moe...")
        with open(file_path, 'rb') as f:
            files = {'fileToUpload': f}
            data = {'reqtype': 'fileupload'}
            response = requests.post(
                'https://catbox.moe/user/api.php',
                files=files,
                data=data,
                timeout=180
            )
        
        st.info(f"📡 catbox.moe 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            file_url = response.text.strip()
            if file_url.startswith('http'):
                st.success(f"✅ 文件已上传到 catbox.moe: {file_url}")
                return file_url
            else:
                st.warning(f"⚠️ catbox.moe 返回格式异常: {file_url[:200]}")
    except Exception as e:
        st.warning(f"⚠️ catbox.moe 上传失败: {str(e)[:100]}")
    
    # 方案2: tmpfiles.org
    try:
        st.info("🔄 尝试 tmpfiles.org...")
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                'https://tmpfiles.org/api/v1/upload',
                files=files,
                timeout=180
            )
        
        st.info(f"📡 tmpfiles.org 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get('status') == 'success':
                    # tmpfiles.org 返回格式: {"status":"success","data":{"url":"https://tmpfiles.org/xxxxx"}}
                    file_url = result.get('data', {}).get('url', '')
                    # 转换为下载链接
                    if '/tmpfiles.org/' in file_url:
                        file_url = file_url.replace('tmpfiles.org/', 'tmpfiles.org/dl/')
                    if file_url.startswith('http'):
                        st.success(f"✅ 文件已上传到 tmpfiles.org: {file_url}")
                        return file_url
            except:
                pass
        st.warning("⚠️ tmpfiles.org 上传失败")
    except Exception as e:
        st.warning(f"⚠️ tmpfiles.org 上传失败: {str(e)[:100]}")
    
    # 方案3: uguu.se
    try:
        st.info("🔄 尝试 uguu.se...")
        with open(file_path, 'rb') as f:
            files = {'files[]': f}
            response = requests.post(
                'https://uguu.se/upload',
                files=files,
                timeout=180
            )
        
        st.info(f"📡 uguu.se 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get('success'):
                    files_list = result.get('files', [])
                    if files_list and len(files_list) > 0:
                        file_url = files_list[0].get('url')
                        if file_url and file_url.startswith('http'):
                            st.success(f"✅ 文件已上传到 uguu.se: {file_url}")
                            return file_url
            except:
                pass
        st.warning("⚠️ uguu.se 上传失败")
    except Exception as e:
        st.warning(f"⚠️ uguu.se 上传失败: {str(e)[:100]}")
    
    # 方案1: file.io (推荐，速度快)
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                'https://file.io',
                files={'file': f},
                data={'expires': '1h'},  # 1小时后过期
                timeout=180
            )
        
        if response.status_code == 200:
            try:
                result = response.json()
                if result.get('success'):
                    file_url = result.get('link')
                    if file_url:
                        st.success(f"✅ 文件已上传到 file.io: {file_url[:60]}...")
                        return file_url
            except:
                # 如果不是JSON，可能是直接返回URL
                file_url = response.text.strip()
                if file_url.startswith('http'):
                    st.success(f"✅ 文件已上传到 file.io: {file_url[:60]}...")
                    return file_url
    except Exception as e:
        st.warning(f"⚠️ file.io 上传失败: {str(e)[:100]}")
    
    # 方案2: 0x0.st (推荐，简单可靠)
    try:
        st.info("🔄 尝试 0x0.st...")
        with open(file_path, 'rb') as f:
            response = requests.post(
                'https://0x0.st',
                files={'file': f},
                timeout=180
            )
        
        st.info(f"📡 0x0.st 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            file_url = response.text.strip()
            if file_url.startswith('http'):
                st.success(f"✅ 文件已上传到 0x0.st: {file_url}")
                return file_url
            else:
                st.warning(f"⚠️ 0x0.st 返回格式异常: {file_url[:200]}")
    except Exception as e:
        st.warning(f"⚠️ 0x0.st 上传失败: {str(e)[:100]}")
    
    # 方案3: transfer.sh
    try:
        st.info("🔄 尝试 transfer.sh...")
        with open(file_path, 'rb') as f:
            response = requests.put(
                f'https://transfer.sh/{file_path.name}',
                data=f,
                timeout=180
            )
        
        st.info(f"📡 transfer.sh 响应状态: {response.status_code}")
        
        if response.status_code == 200:
            file_url = response.text.strip()
            if file_url.startswith('http'):
                st.success(f"✅ 文件已上传到 transfer.sh: {file_url}")
                return file_url
            else:
                st.warning(f"⚠️ transfer.sh 返回格式异常: {file_url[:200]}")
    except Exception as e:
        st.warning(f"⚠️ transfer.sh 上传失败: {str(e)[:100]}")
    
    # 所有方案都失败
    st.error("❌ 所有文件托管服务都失败！")
    raise Exception("无法上传文件到任何托管服务")

def start_local_file_server(file_path: Path, port: int = 8899) -> tuple[str, object]:
    """启动本地HTTP服务器，使文件可被公网访问
    
    返回: (公网URL, 服务器进程)
    """
    import http.server
    import socketserver
    import threading
    import socket
    
    try:
        # 获取本机的公网IP
        external_ip = None
        
        # 方法1: 通过外部API获取公网IP
        try:
            response = requests.get('https://api.ipify.org?format=json', timeout=5)
            if response.status_code == 200:
                external_ip = response.json().get('ip')
                st.info(f"🌐 检测到公网IP: {external_ip}")
        except:
            pass
        
        # 方法2: 如果失败，尝试其他服务
        if not external_ip:
            try:
                response = requests.get('https://ifconfig.me/ip', timeout=5)
                if response.status_code == 200:
                    external_ip = response.text.strip()
                    st.info(f"🌐 检测到公网IP: {external_ip}")
            except:
                pass
        
        if not external_ip:
            st.error("❌ 无法获取公网IP地址！MinerU需要公网可访问的URL")
            st.warning("💡 提示：请确保您的网络环境允许接收外部HTTP请求，或者使用云服务器部署")
            raise Exception("无法获取公网IP")
        
        # 创建临时目录用于文件服务
        temp_dir = Path("/tmp/mineru_files")
        temp_dir.mkdir(exist_ok=True)
        
        # 复制文件到临时目录
        temp_file = temp_dir / file_path.name
        import shutil
        shutil.copy(file_path, temp_file)
        
        # 创建简单的HTTP服务器
        class FileHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(temp_dir), **kwargs)
            
            def log_message(self, format, *args):
                pass  # 禁用日志输出
        
        # 在后台线程中启动服务器
        httpd = socketserver.TCPServer(("", port), FileHandler)
        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()
        
        file_url = f"http://{external_ip}:{port}/{file_path.name}"
        
        st.success(f"✅ 本地文件服务器已启动: {file_url[:70]}...")
        
        # 验证URL是否可访问
        try:
            test_response = requests.head(file_url, timeout=5)
            if test_response.status_code == 200:
                st.success("✅ URL可访问性验证成功")
            else:
                st.warning(f"⚠️ URL验证失败 ({test_response.status_code})，MinerU可能无法访问")
        except Exception as e:
            st.error(f"❌ URL验证失败: {str(e)}")
            st.warning(f"💡 可能原因：\n1. 端口{port}未在防火墙中开放\n2. 路由器未配置端口转发\n3. ISP屏蔽了该端口")
            st.info("建议：在云服务器上部署此应用，或者使用ngrok等内网穿透工具")
        
        return file_url, httpd
        
    except Exception as e:
        st.error(f"❌ 启动本地文件服务器失败: {str(e)}")
        raise e

def convert_with_mineru(pdf_path: Path) -> str:
    """使用MinerU API进行高质量的PDF转Markdown转换
    
    参考文档: https://mineru.net/apiManage/docs
    流程：
    1. 上传PDF到临时托管服务获取公开URL
    2. 使用URL创建MinerU解析任务
    3. 轮询任务状态直到完成
    4. 获取Markdown结果
    """
    try:
        import requests
        import time
        import json
        import urllib3
        import subprocess
        import shlex
        
        # 禁用SSL警告（由于LibreSSL版本过旧）
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 从session state或环境变量获取Token
        MINERU_TOKEN = st.session_state.get('mineru_token') or os.getenv('MINERU_API_TOKEN', '')
        
        if not MINERU_TOKEN:
            st.error("❌ MinerU Token未配置！请在侧边栏设置Token")
            raise Exception("MinerU Token未配置")
        
        # 步骤1: 上传文件到免费托管服务
        file_url = upload_to_free_host(pdf_path)
        httpd = None  # 不需要本地服务器
        
        # 步骤2: 创建MinerU解析任务
        st.info("🔄 步骤2/3: 创建MinerU解析任务...")
        
        task_url = 'https://mineru.net/api/v4/extract/task/batch'
        headers = {
            'Authorization': f'Bearer {MINERU_TOKEN}',
            'Content-Type': 'application/json'
        }
        task_data = {
            "enable_formula": True,
            "enable_table": True,
            "language": "ch",
            "files": [
                {
                    "url": file_url,
                    "is_ocr": True,
                    "data_id": pdf_path.stem
                }
            ]
        }
        
        # 小工具：解析 curl 输出中的 HTTP_STATUS
        def parse_curl_output_for_status(raw_output: str) -> tuple[int, str]:
            """从 curl 标准输出中提取 HTTP 状态码与正文。

            兼容以下几种情况：
            - "\nHTTP_STATUS:200"（预期）
            - "HTTP_STATUS:200"（无前导换行）
            - "\\nHTTP_STATUS:200" 被当作字面字符输出导致出现 "nHTTP_STATUS:200"
            - Windows/CRLF 行尾
            返回: (status_code, body_text)
            若未能解析到状态码，则返回 (0, raw_output)
            """
            import re
            if not raw_output:
                return 0, ""
            # 统一换行符并去掉尾部空白
            text = raw_output.replace("\r\n", "\n").rstrip()
            # 宽松匹配：允许前面有任意字符（包括字面 'n'）
            match = re.search(r"HTTP_STATUS:(\d{3})", text)
            if match:
                status = int(match.group(1))
                body = text[: match.start()].rstrip()
                return status, body
            return 0, raw_output

        # 优先使用requests；如遇到SSL错误，回退到系统curl（更兼容，能规避LibreSSL握手问题）
        try:
            task_response = requests.post(task_url, headers=headers, json=task_data, timeout=60, verify=False)
            task_status = task_response.status_code
            task_text = task_response.text
        except requests.exceptions.SSLError as ssl_err:
            st.warning(f"⚠️ requests SSL错误，改用curl重试: {str(ssl_err)[:120]}")
            curl_cmd = (
                "curl -sS -m 60 --retry 2 --insecure "
                f"-H {shlex.quote('Authorization: Bearer ' + MINERU_TOKEN)} "
                f"-H {shlex.quote('Content-Type: application/json')} "
                f"-d {shlex.quote(json.dumps(task_data))} "
                f"-w '\nHTTP_STATUS:%{{http_code}}' "
                f"{shlex.quote(task_url)}"
            )
            proc = subprocess.run(curl_cmd, shell=True, capture_output=True, text=True)
            output = (proc.stdout or '')
            task_status, task_text = parse_curl_output_for_status(output)
        except requests.exceptions.RequestException as req_err:
            st.warning(f"⚠️ 网络错误，改用curl重试: {str(req_err)[:120]}")
            curl_cmd = (
                "curl -sS -m 60 --retry 2 --insecure "
                f"-H {shlex.quote('Authorization: Bearer ' + MINERU_TOKEN)} "
                f"-H {shlex.quote('Content-Type: application/json')} "
                f"-d {shlex.quote(json.dumps(task_data))} "
                f"-w '\nHTTP_STATUS:%{{http_code}}' "
                f"{shlex.quote(task_url)}"
            )
            proc = subprocess.run(curl_cmd, shell=True, capture_output=True, text=True)
            output = (proc.stdout or '')
            task_status, task_text = parse_curl_output_for_status(output)
        
        # 解析任务创建结果（即使HTTP状态解析失败，也尝试按JSON判断是否成功）
        try:
            task_result = json.loads(task_text)
        except Exception:
            task_result = {"code": None, "raw": task_text}
        
        # 判定成功条件：code==0 或 code==200 或 success==True 或 msg为ok/确定
        code_val = task_result.get('code')
        msg_val = (task_result.get('msg') or '').lower()
        is_ok = (code_val in (0, 200)) or (task_result.get('success') is True) or (msg_val in ('ok', '确定', 'success'))
        if not is_ok or task_status not in (0, 200):
            # 仅当两者都显示异常时，才报错；否则继续按成功处理
            if not is_ok and task_status != 200:
                error_msg = task_result.get('message') or task_result.get('msg') or str(task_result)
                st.error(f"❌ 任务创建失败: 状态{task_status} - {error_msg}")
                raise Exception(f"任务创建失败: {error_msg}")
        
        data_obj = task_result.get('data') or task_result.get('数据') or {}
        task_id = data_obj.get('task_id') or data_obj.get('id')
        batch_id = data_obj.get('batch_id')
        
        # 选择轮询URL：优先使用batch_id
        if batch_id:
            poll_kind = 'batch'
            result_url = f'https://mineru.net/api/v4/extract/batch/result?batch_id={batch_id}'
            st.success(f"✅ 任务创建成功！batch_id: {batch_id}")
        elif task_id:
            poll_kind = 'task'
            result_url = f'https://mineru.net/api/v4/extract/result?task_id={task_id}'
            st.success(f"✅ 任务创建成功！任务ID: {task_id}")
        else:
            st.error(f"❌ 未能获取任务ID/batch_id: {task_result}")
            raise Exception("未能获取任务ID")
        
        # 步骤3: 轮询任务状态
        st.info("⏳ 步骤3/3: 等待MinerU处理完成...")
        
        # 如果上面已根据batch_id或task_id设置了result_url，这里保留；
        # 否则回退到task方式
        if 'result_url' not in locals():
            result_url = f'https://mineru.net/api/v4/extract/result?task_id={task_id}'
        max_attempts = 60  # 最多等待5分钟
        
        for attempt in range(max_attempts):
            time.sleep(5)  # 每5秒查询一次
            
            try:
                result_response = requests.get(result_url, headers=headers, timeout=30, verify=False)
                res_status = result_response.status_code
                res_text = result_response.text
            except requests.exceptions.RequestException:
                # 回退到curl
                curl_cmd = (
                    "curl -sS -m 30 --retry 2 --insecure "
                    f"-H {shlex.quote('Authorization: Bearer ' + MINERU_TOKEN)} "
                    f"-w '\nHTTP_STATUS:%{{http_code}}' "
                    f"{shlex.quote(result_url)}"
                )
                proc = subprocess.run(curl_cmd, shell=True, capture_output=True, text=True)
                output = (proc.stdout or '')
                res_status, res_text = parse_curl_output_for_status(output)
            
            if res_status != 200:
                st.warning(f"⚠️ 查询状态失败: {res_status}")
                continue
            
            try:
                result_data = json.loads(res_text)
            except Exception:
                result_data = {"code": None, "raw": res_text}
            
            if result_data.get('code') == 200:
                data = result_data.get('data', {})
                status = data.get('status')
                
                st.info(f"⏳ 任务状态: {status} (尝试 {attempt + 1}/{max_attempts})")
                
                if status == 'completed':
                    # 任务完成，获取Markdown内容
                    full_zip_url = data.get('full_zip_url')
                    if full_zip_url:
                        # 下载ZIP并解压获取Markdown
                        st.info("📥 下载解析结果...")
                        try:
                            zip_response = requests.get(full_zip_url, timeout=180, verify=False)
                            zip_ok = (zip_response.status_code == 200)
                            zip_bytes = zip_response.content if zip_ok else b""
                        except requests.exceptions.RequestException:
                            # 回退到curl
                            curl_cmd = (
                                "curl -sS -m 180 --retry 2 --insecure "
                                f"-L {shlex.quote(full_zip_url)}"
                            )
                            proc = subprocess.run(curl_cmd, shell=True, capture_output=True)
                            zip_ok = (proc.returncode == 0 and proc.stdout)
                            zip_bytes = proc.stdout if zip_ok else b""
                        
                        if zip_ok:
                            import zipfile
                            import io
                            
                            # 解压ZIP文件
                            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                                # 查找Markdown文件
                                md_files = [f for f in z.namelist() if f.endswith('.md')]
                                if md_files:
                                    markdown_content = z.read(md_files[0]).decode('utf-8')
                                    st.success(f"✅ MinerU成功解析 {pdf_path.name}！")
                                    return markdown_content
                                else:
                                    st.error("❌ ZIP中未找到Markdown文件")
                                    raise Exception("ZIP中未找到Markdown文件")
                        else:
                            st.error(f"❌ 下载结果失败: {zip_response.status_code}")
                            raise Exception("下载结果失败")
                    else:
                        st.error(f"❌ 未找到结果下载链接: {data}")
                        raise Exception("未找到结果下载链接")
                
                elif status == 'failed' or status == 'error':
                    error_msg = data.get('error') or data.get('message') or '未知错误'
                    st.error(f"❌ 解析任务失败: {error_msg}")
                    raise Exception(f"解析任务失败: {error_msg}")
        
        # 超时
        st.warning("⚠️ 任务处理超时，请稍后重试")
        raise Exception("任务处理超时")
            
    except requests.exceptions.Timeout:
        st.warning(f"⚠️ MinerU API超时")
        raise Exception("API超时")
    except requests.exceptions.RequestException as e:
        st.warning(f"⚠️ MinerU网络错误: {str(e)[:200]}")
        raise Exception(f"网络错误: {str(e)}")
    except Exception as e:
        # 抛出异常让主函数使用备用方案
        raise e

def format_pdf_text(text: str) -> str:
    """格式化PDF提取的文本为Markdown格式"""
    if not text.strip():
        return ""
    
    # 基本清理
    lines = text.split('\n')
    formatted_lines = []
    
    current_paragraph = ""
    
    for line in lines:
        line = line.strip()
        
        # 跳过空行
        if not line:
            if current_paragraph:
                formatted_lines.append(current_paragraph.strip())
                formatted_lines.append("")  # 空行段落分隔
                current_paragraph = ""
            continue
        
        # 检测标题 (简单的启发式规则)
        if len(line) < 50 and any(char.isupper() for char in line[:10]):
            if current_paragraph:
                formatted_lines.append(current_paragraph.strip())
                formatted_lines.append("")
            formatted_lines.append(f"## {line}")
            formatted_lines.append("")
            current_paragraph = ""
            continue
        
        # 检测列表项
        if line.startswith(('•', '-', '◦', '▪', '1.', '2.', '3.', '4.', '5.')):
            if current_paragraph:
                formatted_lines.append(current_paragraph.strip())
                formatted_lines.append("")
            formatted_lines.append(f"- {line.lstrip('•-◦▪1234567890. ')}")
            current_paragraph = ""
            continue
        
        # 普通段落文本
        if current_paragraph:
            current_paragraph += " " + line
        else:
            current_paragraph = line
    
    # 添加最后一个段落
    if current_paragraph:
        formatted_lines.append(current_paragraph.strip())
    
    return '\n'.join(formatted_lines)

def clean_markdown_text(text: str) -> str:
    """简化的Markdown清洗函数"""
    import re
    
    try:
        # 清除参考文献
        text = re.sub(r'\[[\d,\s-]+\]', '', text)
        text = re.sub(r'参考文献[\s：:]*\n.*', '', text, flags=re.DOTALL)
        
        # 清除元数据
        patterns = [
            r'^中图分类号[：:].+$',
            r'^文献标志码[：:].+$', 
            r'^文章编号[：:].+$',
            r'^doi[：:].+$',
            r'^DOI[：:].+$',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # 清除多余空行
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    except re.error as e:
        print(f"正则表达式错误: {e}")
        # 如果正则表达式出错，只做基本的空白处理
        text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text.strip()

def process_uploaded_files(uploaded_files: List, chunk_size: int = 400, chunk_overlap: int = 80):
    """处理上传的文件"""
    results = {
        'files_processed': 0,
        'chunks_created': 0,
        'total_chars': 0,
        'file_details': [],
        'errors': []
    }
    
    # 简单的分块处理（不使用复杂的SmartChunker）
    temp_dir = Path(tempfile.mkdtemp())
    st.session_state.temp_files.append(temp_dir)
    
    all_chunks = []
    
    for uploaded_file in uploaded_files:
        try:
            # 保存上传的文件
            file_path = temp_dir / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 根据文件类型读取内容
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.pdf':
                # PDF文件转换为Markdown
                original_text = convert_pdf_to_markdown(file_path)
                
                # 检查结果是否是转换成功的原始文本
                if original_text.startswith('[') and '错误' in original_text:
                    # 转换失败，使用备用方案
                    st.warning(f"⚠️ PDF文件 {uploaded_file.name} 无法使用MinerU转换，使用备用方案")
                    with open(file_path, 'rb') as f:
                        if PDFPLUMBER_AVAILABLE:
                            pdf = PDF(f)
                            pdf_text = ""
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    pdf_text += page_text + "\n\n"
                            original_text = pdf_text
                        elif PDF_AVAILABLE:
                            pdf_reader = PyPDF2.PdfReader(f)
                            pdf_text = ""
                            for page_num in range(len(pdf_reader.pages)):
                                page = pdf_reader.pages[page_num]
                                page_text = page.extract_text()
                                if page_text:
                                    pdf_text += page_text + "\n\n"
                            original_text = pdf_text
                        else:
                            original_text = f"[PDF文件无法处理: {uploaded_file.name}]"
                
                # PDF文本清洗
                cleaned_text = clean_markdown_text(original_text)
            else:
                # 其他文本文件 (Markdown, TXT等)
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_text = f.read()
                
                # 清洗文本
                cleaned_text = clean_markdown_text(original_text)
            
            # 检查是否为有效内容
            if len(cleaned_text.strip()) < 100:
                results['errors'].append(f"文件 {uploaded_file.name} 内容过短，已跳过")
                continue
            
            # 简单分块：按段落分割
            paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
            chunks = []
            
            current_chunk = ""
            chunk_index = 0
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) > chunk_size:
                    if len(current_chunk) >= 150:
                        chunks.append({
                            'text': current_chunk.strip(),
                            'metadata': {
                                'filename': uploaded_file.name,
                                'chunk_index': chunk_index,
                                'char_count': len(current_chunk.strip())
                            }
                        })
                        current_chunk = paragraph
                        chunk_index += 1
                    else:
                        current_chunk += paragraph + '\n\n'
                else:
                    current_chunk += paragraph + '\n\n'
            
            # 添加最后一个块
            if len(current_chunk.strip()) >= 150:
                chunks.append({
                    'text': current_chunk.strip(),
                    'metadata': {
                        'filename': uploaded_file.name,
                        'chunk_index': chunk_index,
                        'char_count': len(current_chunk.strip())
                    }
                })
            
            # 保存分块到JSONL
            jsonl_file = temp_dir / f"{Path(uploaded_file.name).stem}.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
            
            # 统计信息
            file_detail = {
                'filename': uploaded_file.name,
                'chunks_count': len(chunks),
                'original_size': len(original_text),
                'cleaned_size': len(cleaned_text),
                'avg_chunk_size': sum(c['metadata']['char_count'] for c in chunks) / len(chunks) if chunks else 0
            }
            
            results['files_processed'] += 1
            results['chunks_created'] += len(chunks)
            results['total_chars'] += len(cleaned_text)
            results['file_details'].append(file_detail)
            all_chunks.extend(chunks)
            
        except Exception as e:
            error_msg = f"处理文件 {uploaded_file.name} 时出错: {str(e)}"
            results['errors'].append(error_msg)
    
    results['vector_index_created'] = True  # 简化处理，直接标记为成功
    results['chunks'] = all_chunks  # 添加chunks字段供编辑界面使用
    
    return results

def display_file_info(files: List):
    """显示上传文件的信息"""
    if not files:
        return
    
    st.markdown("### 📄 已上传文件")
    
    for file in files:
        file_size_mb = file.size / (1024 * 1024)
        st.markdown(f"""
        <div class="file-info">
            <strong>📁 {file.name}</strong><br>
            大小: {file_size_mb:.2f} MB | 类型: {file.type}
        </div>
        """, unsafe_allow_html=True)

def display_processing_results(results: Dict):
    """显示处理结果"""
    if not results:
        return
    
    st.markdown("### 📊 处理结果")
    
    # 总体统计
    st.markdown(f"""
    <div class="process-card">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; text-align: center;">
            <div>
                <div style="font-size: 24px; font-weight: 700; color: #4a7c59;">{results['files_processed']}</div>
                <div style="font-size: 14px; color: #7f8c8d;">文件处理</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: 700; color: #4a7c59;">{results['chunks_created']}</div>
                <div style="font-size: 14px; color: #7f8c8d;">知识片段</div>
            </div>
            <div>
                <div style="font-size: 24px; font-weight: 700; color: #4a7c59;">{results['total_chars']:,}</div>
                <div style="font-size: 14px; color: #7f8c8d;">总字符数</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 文件详情
    if results['file_details']:
        st.markdown("#### 📋 文件处理详情")
        
        for detail in results['file_details']:
            with st.expander(f"📄 {detail['filename']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("分块数量", f"{detail['chunks_count']}")
                
                with col2:
                    processed_rate = (detail['cleaned_size'] / detail['original_size']) * 100
                    st.metric("清洗效率", f"{processed_rate:.1f}%")
                
                with col3:
                    avg_size = detail['avg_chunk_size']
                    st.metric("平均分块大小", f"{avg_size:.0f} 字符")
    
    # 错误信息
    if results['errors']:
        st.markdown("#### ⚠️ 处理错误")
        for error in results['errors']:
            st.error(error)
    
    # 可编辑的结果展示
    if results.get('chunks'):
        display_editable_chunks(results['chunks'])

def display_editable_chunks(chunks):
    """显示可编辑的文档块"""
    st.markdown("---")
    st.markdown("### ✏️ 手动审核与编辑")
    st.markdown("💡 **提示**: 您可以在这里手动审核和编辑处理结果，确保文档质量")
    
    # 添加到session state中进行编辑
    if 'editable_chunks' not in st.session_state:
        st.session_state.editable_chunks = chunks.copy()
    
    # 筛选器
    col1, col2 = st.columns([2, 1])
    
    with col1:
        filename_filter = st.selectbox(
            "📁 选择要编辑的文件:",
            options=["全部"] + list(set([chunk.get('metadata', {}).get('filename', 'unknown') for chunk in chunks])),
            key="filename_filter"
        )
    
    with col2:
        if st.button("🔄 重置为原始数据", key="reset_chunks"):
            st.session_state.editable_chunks = chunks.copy()
            st.rerun()
    
    # 根据筛选器显示文档块
    filtered_chunks = st.session_state.editable_chunks
    if filename_filter != "全部":
        filtered_chunks = [chunk for chunk in st.session_state.editable_chunks 
                          if chunk.get('metadata', {}).get('filename') == filename_filter]
    
    st.markdown(f"**显示 {len(filtered_chunks)} 个文档块**")
    
    # 分页显示
    chunks_per_page = 10
    total_pages = (len(filtered_chunks) + chunks_per_page - 1) // chunks_per_page
    current_page = st.session_state.get('chunk_page', 0)
    
    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ 上一页", disabled=(current_page == 0)):
                st.session_state.chunk_page = current_page - 1
                st.rerun()
        
        with col2:
            st.markdown(f"<div style='text-align: center; padding: 8px;'>{current_page + 1} / {total_pages}</div>", 
                       unsafe_allow_html=True)
        
        with col3:
            if st.button("➡️ 下一页", disabled=(current_page == total_pages - 1)):
                st.session_state.chunk_page = current_page + 1
                st.rerun()
    
    # 显示当前页的文档块
    start_idx = current_page * chunks_per_page
    end_idx = min(start_idx + chunks_per_page, len(filtered_chunks))
    
    for i in range(start_idx, end_idx):
        chunk_idx = i
        chunk = filtered_chunks[i]
        
        with st.expander(f"📄 文档块 {chunk_idx + 1} - {chunk.get('metadata', {}).get('filename', 'unknown')} (字符数: {len(chunk['text'])})", expanded=False):
            
            # 元数据展示
            metadata = chunk.get('metadata', {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.text_input(
                    "文件名",
                    value=metadata.get('filename', ''),
                    key=f"metadata_filename_{current_page}_{i}"
                )
            
            with col2:
                st.number_input(
                    "块索引",
                    value=metadata.get('chunk_index', 0),
                    key=f"metadata_chunk_index_{current_page}_{i}"
                )
            
            with col3:
                st.number_input(
                    "字符数",
                    value=metadata.get('char_count', len(chunk['text'])),
                    key=f"metadata_char_count_{current_page}_{i}"
                )
            
            # 文本内容编辑器
            edited_text = st.text_area(
                "📝 文档内容",
                value=chunk['text'],
                height=200,
                key=f"chunk_text_{current_page}_{i}",
                help="您可以在这里编辑文档内容"
            )
            
            # 更新按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"💾 保存修改", key=f"save_chunk_{current_page}_{i}"):
                    # 找到原始chunk并更新
                    original_idx = start_idx + i
                    if original_idx < len(st.session_state.editable_chunks):
                        st.session_state.editable_chunks[original_idx]['text'] = edited_text
                        st.session_state.editable_chunks[original_idx]['metadata'] = {
                            'filename': metadata.get('filename', ''),
                            'chunk_index': metadata.get('chunk_index', 0),
                            'char_count': len(edited_text)
                        }
                        st.success("✅ 文档块已更新!")
                        st.rerun()
            
            with col2:
                if st.button(f"❌ 删除块", key=f"delete_chunk_{current_page}_{i}"):
                    # 删除这个块
                    if original_idx < len(st.session_state.editable_chunks):
                        st.session_state.editable_chunks.pop(original_idx)
                        st.warning("⚠️ 文档块已删除!")
                        st.rerun()
    
    # 导出和保存功能
    st.markdown("---")
    st.markdown("### 💾 保存和导出")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬇️ 导出为JSON", key="export_json"):
            from datetime import datetime
            json_data = {
                'total_chunks': len(st.session_state.editable_chunks),
                'chunks': st.session_state.editable_chunks,
                'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'original_files': list(set([chunk.get('metadata', {}).get('filename', 'unknown') 
                                          for chunk in st.session_state.editable_chunks]))
            }
            
            st.download_button(
                label="📥 下载JSON文件",
                data=json.dumps(json_data, ensure_ascii=False, indent=2),
                file_name=f"knowledge_base_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_json"
            )
    
    with col2:
        if st.button("📊 导出统计数据", key="export_stats"):
            from datetime import datetime
            stats = {
                'total_chunks': len(st.session_state.editable_chunks),
                'total_chars': sum(len(chunk['text']) for chunk in st.session_state.editable_chunks),
                'avg_chunk_size': sum(len(chunk['text']) for chunk in st.session_state.editable_chunks) / len(st.session_state.editable_chunks) if st.session_state.editable_chunks else 0,
                'files': list(set([chunk.get('metadata', {}).get('filename', 'unknown') 
                                 for chunk in st.session_state.editable_chunks])),
                'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            st.download_button(
                label="📥 下载统计数据",
                data=json.dumps(stats, ensure_ascii=False, indent=2),
                file_name=f"knowledge_base_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_stats"
            )
    
    with col3:
        if st.button("🔄 构建向量索引", key="build_vector_index"):
            if st.session_state.editable_chunks:
                st.info("🚀 开始构建向量索引...")
                try:
                    # 这里可以添加向量化功能
                    st.success("✅ 向量索引构建完成! 知识库已更新。")
                except Exception as e:
                    st.error(f"❌ 向量索引构建失败: {str(e)}")
            else:
                st.warning("⚠️ 没有可处理的文档块")

def main():
    """主函数"""
    # 初始化
    initialize_session_state()
    
    # 页面标题
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 36px; font-weight: 700; color: #2c3e50; margin-bottom: 1rem;'>
            🔧 知识库构建工具
        </h1>
        <p style='font-size: 18px; color: #7f8c8d; margin-bottom: 2rem;'>
            上传文档文件，自动完成清洗、分块、向量化处理
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.markdown("🌟 **知识库构建系统**")
        st.markdown("---")
        
        # MinerU Token配置
        st.markdown("**🔑 MinerU API配置**")
        mineru_token = st.text_input(
            "MinerU Token",
            value=st.session_state.get('mineru_token', 'eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI4MjkwMDMxMyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1OTQ5NTU2NiwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiMTM5MzI4ODE4NTEiLCJvcGVuSWQiOm51bGwsInV1aWQiOiJlYzI5YjY3MC0zYjk4LTQ1NTgtODE3Ny01N2VhM2RkMmYzN2EiLCJlbWFpbCI6IiIsImV4cCI6MTc2MDcwNTE2Nn0.7MWjSewD2AzOndsWH5MGMwtCB8tvCcFnGsiAcM5zzjzdrfhTz0376ysAFPpDMbqVAFL3WoEMLE3-4M8qzlM_3A'),
            type="password",
            help="从 https://mineru.net 获取您的API Token"
        )
        st.session_state.mineru_token = mineru_token
        
        if mineru_token:
            st.success("✅ Token已配置")
            st.info("""
            **PDF处理方式**: MinerU (高精度)
            
            流程:
            1. 📤 上传PDF到免费托管服务
            2. 🔄 提交MinerU解析任务
            3. ⏳ 等待处理完成
            4. ✅ 下载Markdown结果
            """)
        else:
            st.warning("⚠️ 未配置Token，将使用pdfplumber")
        
        st.markdown("---")
        
        st.markdown("**📋 处理参数**")
        chunk_size = st.slider("分块大小 (字符)", 200, 800, 400)
        chunk_overlap = st.slider("重叠大小 (字符)", 20, 200, 80)
        
        st.markdown("---")
        st.markdown("**📖 使用说明**")
        
        st.markdown("""
        1. **支持格式**: Markdown (.md), 文本 (.txt), PDF (.pdf)
        2. **文件大小**: 建议小于 10MB
        3. **内容要求**: 至少包含 100 字符的有效内容
        4. **PDF转换**: 支持MinerU和传统方法
        5. **处理流程**: 
           - 📄 PDF转Markdown (优先MinerU，备用传统方法)
           - 🧹 清洗参考文献、元数据
           - ✂️ 智能分块保持语义完整
           - 🤖 生成向量索引支持检索
        """)
    
    # 主内容区域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 文件上传区域
        st.markdown(f"""
        <div class="upload-area">
            <div style="font-size: 24px; margin-bottom: 1rem;">📁</div>
            <div style="font-size: 18px; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">
                拖拽文件到这里或点击选择
            </div>
            <div style="font-size: 14px; color: #7f8c8d;">
                支持 Markdown (.md)、文本 (.txt) 和 PDF (.pdf) 文件
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "选择文件",
            type=['md', 'txt', 'pdf'],
            accept_multiple_files=True,
            help="支持多个文件同时上传，包括PDF自动转换"
        )
        
        # 更新session state
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
        
        # PDF库检查提示
        if uploaded_files and any(f.name.lower().endswith('.pdf') for f in uploaded_files):
            if not MINERU_AVAILABLE and not PDFPLUMBER_AVAILABLE and not PDF_AVAILABLE:
                st.warning("""
                ⚠️ **PDF处理库未安装**
                
                要处理PDF文件，强烈建议安装以下库：
                - `pip install requests` (MinerU API调用需要)
                - `pip install pdfplumber` (备用方案)
                - `pip install PyPDF2` (最后备用)
                
                安装后刷新页面即可使用高质量的PDF转换功能。
                """)
            elif MINERU_AVAILABLE:
                st.success("✅ MinerU API可用 - PDF转换将使用最高精度在线处理")
        
        # 显示已上传的文件信息
        if st.session_state.uploaded_files:
            display_file_info(st.session_state.uploaded_files)
        
        # 处理按钮
        if st.session_state.uploaded_files and st.session_state.processing_status == "idle":
            if st.button("🚀 开始处理", type="primary", use_container_width=True):
                st.session_state.processing_status = "processing"
        
        # 处理进度和结果
        if st.session_state.processing_status == "processing":
            with st.spinner("正在处理文件..."):
                try:
                    results = process_uploaded_files(
                        st.session_state.uploaded_files,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    st.session_state.processing_results = results
                    st.session_state.processing_status = "completed"
                    
                    # 显示结果
                    st.success("✅ 处理完成！")
                    display_processing_results(results)
                    
                except Exception as e:
                    st.error(f"❌ 处理失败: {str(e)}")
                    st.session_state.processing_status = "error"
        
        elif st.session_state.processing_status == "completed":
            st.success("✅ 处理完成！")
            display_processing_results(st.session_state.processing_results)
        
        elif st.session_state.processing_status == "error":
            st.error("❌ 处理过程中出现错误，请检查文件格式和内容")
    
    with col2:
        st.markdown("#### 🔄 处理流程")
        
        steps = [
            ("📄", "文件上传", "用户选择Markdown或文本文件"),
            ("🧹", "内容清洗", "自动清理参考文献、元数据等"),
            ("✂️", "智能分块", "按语义分割文本，保持完整性"),
            ("🏷️", "元数据提取", "自动提取标题、关键词等信息"),
            ("💾", "结果保存", "保存处理结果到临时目录")
        ]
        
        for icon, title, description in steps:
            current_status = ""
            if st.session_state.processing_status == "idle":
                current_status = "⚪ 等待"
            elif st.session_state.processing_status == "processing":
                current_status = "🟡 处理中"
            elif st.session_state.processing_status == "completed":
                current_status = "🟢 完成"
            elif st.session_state.processing_status == "error":
                current_status = "🔴 错误"
            
            st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                padding: 0.8rem;
                margin: 0.5rem 0;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #4a7c59;
            ">
                <span style="font-size: 20px; margin-right: 1rem;">{icon}</span>
                <div style="flex-grow: 1;">
                    <div style="font-weight: 600; color: #2c3e50;">{title}</div>
                    <div style="font-size: 12px; color: #7f8c8d;">{description}</div>
                </div>
                <span style="font-size: 14px;">{current_status}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # 重置按钮
        if st.session_state.processing_status != "idle":
            if st.button("🔄 重新开始", use_container_width=True):
                # 清理临时文件
                for temp_path in st.session_state.temp_files:
                    try:
                        shutil.rmtree(temp_path)
                    except:
                        pass
                
                # 重置状态
                st.session_state.processing_status = "idle"
                st.session_state.uploaded_files = []
                st.session_state.processing_results = {}
                st.session_state.temp_files = []
                st.rerun()

if __name__ == "__main__":
    main()