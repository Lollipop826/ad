"""
地理位置服务 - 自动获取当前位置信息
支持多种方式：IP定位、浏览器Geolocation、手动配置
同时提供实时天气信息
"""

import json
import os
import httpx
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

# 配置文件路径
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
LOCATION_CONFIG_FILE = CONFIG_DIR / "deployment.json"


def get_location_from_ip() -> Optional[Dict]:
    """
    通过IP地址获取地理位置（免费API）
    返回: {"province": "广东省", "city": "深圳市", "district": "南山区", ...}
    """
    apis = [
        # 方案1: ip-api.com (免费，无需key)
        {
            "url": "http://ip-api.com/json/?lang=zh-CN",
            "parser": lambda r: {
                "province": r.get("regionName", ""),
                "city": r.get("city", ""),
                "district": "",  # IP定位通常没有区级信息
                "place": "家中",
                "country": r.get("country", ""),
                "lat": r.get("lat"),
                "lon": r.get("lon"),
                "source": "ip-api.com"
            }
        },
        # 方案2: ipinfo.io (备用)
        {
            "url": "https://ipinfo.io/json",
            "parser": lambda r: {
                "province": r.get("region", ""),
                "city": r.get("city", ""),
                "district": "",
                "place": "家中",
                "country": r.get("country", ""),
                "source": "ipinfo.io"
            }
        },
    ]
    
    for api in apis:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(api["url"])
                if response.status_code == 200:
                    data = response.json()
                    location = api["parser"](data)
                    if location.get("city"):  # 至少要有城市信息
                        print(f"[LocationService] ✅ IP定位成功: {location['province']} {location['city']}")
                        return location
        except Exception as e:
            print(f"[LocationService] ⚠️ {api['url']} 失败: {e}")
            continue
    
    return None


def get_location_from_config() -> Optional[Dict]:
    """从配置文件读取位置"""
    try:
        if LOCATION_CONFIG_FILE.exists():
            with open(LOCATION_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("location")
    except Exception as e:
        print(f"[LocationService] ⚠️ 读取配置失败: {e}")
    return None


def save_location_to_config(location: Dict) -> bool:
    """保存位置到配置文件"""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        config = {
            "location": location,
            "updated_at": datetime.now().isoformat(),
            "auto_detected": True
        }
        
        with open(LOCATION_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        
        print(f"[LocationService] 💾 位置已保存到配置文件")
        return True
    except Exception as e:
        print(f"[LocationService] ❌ 保存配置失败: {e}")
        return False


def get_current_location(force_refresh: bool = False) -> Dict:
    """
    获取当前位置（优先级：配置文件 > IP定位 > 默认值）
    
    Args:
        force_refresh: 是否强制重新获取（忽略缓存配置）
    
    Returns:
        位置信息字典
    """
    # 1. 如果不强制刷新，先尝试从配置文件读取
    if not force_refresh:
        cached = get_location_from_config()
        if cached and cached.get("city"):
            print(f"[LocationService] 📍 使用缓存位置: {cached.get('province', '')} {cached.get('city', '')}")
            return cached
    
    # 2. 尝试IP定位
    location = get_location_from_ip()
    if location:
        # 保存到配置文件（缓存）
        save_location_to_config(location)
        return location
    
    # 3. 返回默认值
    print("[LocationService] ⚠️ 无法获取位置，使用默认值")
    return {
        "province": "未知",
        "city": "未知",
        "district": "",
        "place": "家中",
        "source": "default"
    }


def update_location_manually(province: str, city: str, district: str = "", place: str = "") -> Dict:
    """手动更新位置"""
    location = {
        "province": province,
        "city": city,
        "district": district,
        "place": place or "家中",
        "source": "manual"
    }
    save_location_to_config(location)
    return location


def get_weather(city: str = None, lat: float = None, lon: float = None) -> Optional[Dict]:
    """
    获取实时天气信息（使用传入的位置信息）
    
    Args:
        city: 城市名称（中文）- 从IP定位获取
        lat: 纬度 - 从IP定位获取
        lon: 经度 - 从IP定位获取
    
    Returns:
        天气信息字典，失败返回None
    """
    print(f"[WeatherService] 📍 使用位置信息: city={city}, lat={lat}, lon={lon}")
    
    # 尝试1: 使用wttr.in根据城市名获取天气（国内可访问）
    if city:
        try:
            with httpx.Client(timeout=3.0) as client:
                # 使用简化版wttr.in API，支持中文城市名
                url = f"http://wttr.in/{city}?format=%t+%C+%h&lang=zh"  # 温度+天气+湿度
                response = client.get(url)
                if response.status_code == 200:
                    text = response.text.strip()
                    parts = text.split()
                    print(f"[WeatherService] ✅ wttr.in ({city}): {text}")
                    return {
                        "temperature": parts[0] if len(parts) > 0 else "未知",
                        "weather": " ".join(parts[1:-1]) if len(parts) > 2 else "未知",
                        "humidity": parts[-1] if len(parts) > 0 else "未知",
                        "source": "wttr.in"
                    }
        except Exception as e:
            print(f"[WeatherService] ⚠️ wttr.in失败: {e}")
    
    # 尝试2: 使用经纬度获取天气（如果wttr.in失败）
    if lat and lon:
        try:
            with httpx.Client(timeout=3.0) as client:
                # 使用经纬度格式
                url = f"http://wttr.in/{lat},{lon}?format=%t+%C+%h&lang=zh"
                response = client.get(url)
                if response.status_code == 200:
                    text = response.text.strip()
                    parts = text.split()
                    print(f"[WeatherService] ✅ wttr.in (经纬度): {text}")
                    return {
                        "temperature": parts[0] if len(parts) > 0 else "未知",
                        "weather": " ".join(parts[1:-1]) if len(parts) > 2 else "未知",
                        "humidity": parts[-1] if len(parts) > 0 else "未知",
                        "source": "wttr.in-coords"
                    }
        except Exception as e:
            print(f"[WeatherService] ⚠️ wttr.in(coords)失败: {e}")
    
    # 尝试3: 如果都失败，根据月份推断季节性天气
    if city or lat:
        month = datetime.now().month
        if month in [12, 1, 2]:
            weather_desc = "冬季寒冷"
        elif month in [3, 4, 5]:
            weather_desc = "春季温和"
        elif month in [6, 7, 8]:
            weather_desc = "夏季炎热"
        else:
            weather_desc = "秋季凉爽"
        
        print(f"[WeatherService] ⚠️ API失败，使用季节推断: {weather_desc}")
        return {
            "temperature": "实时温度",
            "weather": weather_desc,
            "humidity": "适中",
            "source": "season-inferred"
        }
    
    # 所有方法都失败
    print(f"[WeatherService] ❌ 所有方法均失败")
    return None


# 全局缓存
_cached_location = None
_cached_weather = None
_weather_update_time = None


def get_deployment_location() -> Dict:
    """
    获取部署位置（供其他模块调用的主接口）
    首次调用时自动获取，之后使用缓存
    """
    global _cached_location
    
    if _cached_location is None:
        _cached_location = get_current_location()
    
    return _cached_location


def refresh_location() -> Dict:
    """强制刷新位置"""
    global _cached_location
    _cached_location = get_current_location(force_refresh=True)
    return _cached_location


def get_realtime_context() -> Dict:
    """
    获取完整的实时上下文信息（位置、时间、天气）
    用于MMSE定向力评估
    
    Returns:
        包含位置、时间、天气的完整字典
    """
    global _cached_weather, _weather_update_time
    
    # 1. 获取位置
    location = get_deployment_location()
    
    # 2. 获取当前时间
    now = datetime.now()
    weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
    
    def get_season(month):
        if month in [3, 4, 5]: return "春季"
        elif month in [6, 7, 8]: return "夏季"
        elif month in [9, 10, 11]: return "秋季"
        else: return "冬季"
    
    time_info = {
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "weekday": weekday_names[now.weekday()],
        "season": get_season(now.month),
        "hour": now.hour,
        "minute": now.minute
    }
    
    # 3. 获取天气（缓存10分钟）
    weather = None
    if _cached_weather and _weather_update_time:
        elapsed = (datetime.now() - _weather_update_time).total_seconds()
        if elapsed < 600:  # 10分钟内使用缓存
            weather = _cached_weather
            print(f"[RealtimeContext] 🌤️ 使用缓存天气 ({elapsed:.0f}秒前)")
    
    if not weather:
        # 尝试获取新天气
        city = location.get('city', '')
        lat = location.get('lat')
        lon = location.get('lon')
        
        weather = get_weather(city=city, lat=lat, lon=lon)
        if weather:
            _cached_weather = weather
            _weather_update_time = datetime.now()
        else:
            # 如果获取失败，使用默认值
            weather = {
                "temperature": "未知",
                "weather": "未知",
                "humidity": "未知",
                "source": "unavailable"
            }
    
    # 组合所有信息
    context = {
        "location": location,
        "time": time_info,
        "weather": weather,
        "timestamp": now.isoformat()
    }
    
    return context


# 测试
if __name__ == "__main__":
    print("测试实时上下文获取...")
    ctx = get_realtime_context()
    print(f"完整上下文: {json.dumps(ctx, ensure_ascii=False, indent=2)}")
