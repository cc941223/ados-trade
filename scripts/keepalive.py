#!/usr/bin/env python3
"""IBKR Client Portal Gateway 保活脚本

功能：
1. 每 KEEPALIVE_INTERVAL（默认 50）秒调用一次 /v1/api/tickle，防止 Gateway 会话超时。
2. 每次调用后检查 /v1/api/iserver/auth/status 的 authenticated 字段。
3. 当 authenticated 为 false 时，通过企业微信群机器人 webhook 发送告警，
   内容包含时间戳，并提示需要用手机 IB Key App 确认 2FA 推送。
4. webhook 地址从环境变量 ALERT_WEBHOOK_URL 读取（在 env.list 中配置），不硬编码。

环境变量：
    IBKR_GATEWAY_URL    Gateway 基础地址，默认 https://ibeam:5000（docker-compose 内网服务名）
    ALERT_WEBHOOK_URL   企业微信群机器人 webhook 地址（必填，见 env.list）
    KEEPALIVE_INTERVAL  保活轮询间隔秒数，默认 50
    ALERT_COOLDOWN      告警冷却时间秒数，避免持续掉线时反复刷屏，默认 300
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("keepalive")

GATEWAY_URL = os.environ.get("IBKR_GATEWAY_URL", "https://ibeam:5000").rstrip("/")
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
INTERVAL = float(os.environ.get("KEEPALIVE_INTERVAL", "50"))
ALERT_COOLDOWN = float(os.environ.get("ALERT_COOLDOWN", "300"))

TICKLE_URL = f"{GATEWAY_URL}/v1/api/tickle"
AUTH_STATUS_URL = f"{GATEWAY_URL}/v1/api/iserver/auth/status"

BEIJING_TZ = timezone(timedelta(hours=8))

# IBeam/Gateway 使用自签名证书，只对 Gateway 请求跳过证书校验；
# 企业微信 webhook 走公网 https，仍使用默认证书校验。
_INSECURE_SSL_CONTEXT = ssl.create_default_context()
_INSECURE_SSL_CONTEXT.check_hostname = False
_INSECURE_SSL_CONTEXT.verify_mode = ssl.CERT_NONE


def _request(url: str, method: str = "GET", data: Optional[dict] = None, timeout: float = 10) -> Optional[dict]:
    body = None
    headers = {"Accept": "application/json"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    ctx = _INSECURE_SSL_CONTEXT if url.startswith(GATEWAY_URL) else None
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        raw = resp.read()
        if not raw:
            return None
        return json.loads(raw)


def tickle() -> Optional[dict]:
    """调用 /v1/api/tickle 保活"""
    return _request(TICKLE_URL, method="POST", data={})


def auth_status() -> Optional[dict]:
    """查询 /v1/api/iserver/auth/status"""
    return _request(AUTH_STATUS_URL, method="GET")


def now_str() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S %z")


def send_wecom_alert(text: str) -> None:
    """通过企业微信群机器人 webhook 发送文本告警"""
    if not ALERT_WEBHOOK_URL:
        log.warning("ALERT_WEBHOOK_URL 未配置，跳过告警发送。原始消息: %s", text)
        return

    payload = {"msgtype": "text", "text": {"content": text}}
    try:
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        log.info("已发送企业微信告警")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        log.error("发送企业微信告警失败: %s", e)


def main() -> None:
    if not ALERT_WEBHOOK_URL:
        log.warning(
            "环境变量 ALERT_WEBHOOK_URL 未设置，一旦 Gateway 认证失效将无法发送告警！"
            "请在 env.list 中配置 ALERT_WEBHOOK_URL。"
        )

    log.info("IBKR Gateway 保活脚本启动，目标：%s，轮询间隔：%ss", GATEWAY_URL, INTERVAL)
    last_alert_ts = 0.0

    while True:
        cycle_start = time.monotonic()

        try:
            tickle()
            log.info("tickle 成功")
        except Exception as e:  # noqa: BLE001 - 保活循环需要容错，不能因单次异常退出
            log.error("调用 /v1/api/tickle 失败: %s", e)

        try:
            status = auth_status()
            authenticated = bool(status.get("authenticated")) if status else False
            log.info("认证状态: authenticated=%s", authenticated)

            if not authenticated:
                now = time.monotonic()
                if now - last_alert_ts >= ALERT_COOLDOWN:
                    message = (
                        "【ADOS-Trade 告警】IBKR Gateway 未认证\n"
                        f"时间: {now_str()}\n"
                        "状态: authenticated=false\n"
                        "请尽快打开手机 IB Key App 确认 2FA 推送，完成 Gateway 重新登录认证"
                    )
                    send_wecom_alert(message)
                    last_alert_ts = now
                else:
                    log.info("认证失效但仍在告警冷却期内（%.0fs），跳过重复告警", ALERT_COOLDOWN)
        except Exception as e:  # noqa: BLE001
            log.error("检查 /v1/api/iserver/auth/status 失败: %s", e)

        elapsed = time.monotonic() - cycle_start
        time.sleep(max(0.0, INTERVAL - elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("收到退出信号，保活脚本停止")
        sys.exit(0)
