#!/usr/bin/env python3
"""IB Gateway 连接监控 + 断线告警脚本

⚠️ 改造说明：这个脚本原来是给 IBeam + Client Portal Web API（CPAPI）设计
的——CPAPI 是基于 HTTPS 会话的接口，会话有超时时间，所以需要每隔几十秒
主动调用 `/v1/api/tickle` 保活，再轮询 `/v1/api/iserver/auth/status` 判断
是不是掉线了。现在换成 ib_insync + IB Gateway（TWS API，长连接 socket
协议）之后，这套"定时轮询发现异常"的逻辑不再适用：TWS API 是持久连接，
不需要靠定时请求维持会话，`ib_insync` 本身有连接状态变化的事件回调
（`connectedEvent`/`disconnectedEvent`），所以本脚本改成：

1. 事件驱动：订阅 `ib.disconnectedEvent`，断线事件一触发就立刻发送
   Server酱告警（告警渠道逻辑跟原来完全一样，只是触发方式从"轮询发现
   异常"改成"事件通知异常"）。
2. 兜底心跳：每 `HEARTBEAT_INTERVAL`（默认 60）秒额外检查一次
   `ib.isConnected()`，防止 `disconnectedEvent` 本身也漏报的极端情况
   （例如进程假死但 socket 没有触发标准的断线回调）。
3. 断线后会自动尝试重连（间隔 `RECONNECT_INTERVAL`，默认 30 秒）——这一点
   超出了"只做事件监听 + 心跳兜底"的字面要求，是我加的：没有重连的话，
   一旦断线一次，心跳检查会永远发现"还是断线"，脚本除了一直重复同一条
   告警（哪怕大部分会被冷却期抑制）之外没有别的价值，兜底检查本身也失去意义。
   如果你不想要自动重连（比如担心它跟 VNC 里手动操作的登录状态冲突），
   告诉我，这部分可以去掉，改成纯监控 + 告警、断线后交给人手动处理。

Server酱 SendKey 从环境变量 SERVERCHAN_SENDKEY 读取（在 env.list 中配置），
不硬编码，这部分跟改造前完全一样。

环境变量：
    IBGATEWAY_HOST         IB Gateway 地址，默认 "ibgateway"
                           （docker-compose 内网服务名）
    IBGATEWAY_PORT         TWS API 端口，默认 4002
    IBGATEWAY_CLIENT_ID    TWS API client id，默认 10（故意跟
                           `IBInsyncClient` 默认值 1、
                           `check_ibgateway_connection.py` 默认值 99 都不
                           一样，这三个如果同时连着 Gateway，必须用不同
                           client id，否则后连的会把先连的踢掉）
    SERVERCHAN_SENDKEY     Server酱 SendKey（必填，见 env.list，
                           https://sct.ftqq.com 获取）
    HEARTBEAT_INTERVAL     兜底心跳检查间隔秒数，默认 60（对应"每分钟
                           检查一次"）
    RECONNECT_INTERVAL     断线后重连尝试的间隔秒数，默认 30
    ALERT_COOLDOWN         告警冷却时间秒数，避免持续掉线时反复刷屏，
                           默认 300（跟改造前一样）
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from ib_insync import IB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("keepalive")

HOST = os.environ.get("IBGATEWAY_HOST", "ibgateway")
PORT = int(os.environ.get("IBGATEWAY_PORT", "4002"))
CLIENT_ID = int(os.environ.get("IBGATEWAY_CLIENT_ID", "10"))
SERVERCHAN_SENDKEY = os.environ.get("SERVERCHAN_SENDKEY", "").strip()
HEARTBEAT_INTERVAL = float(os.environ.get("HEARTBEAT_INTERVAL", "60"))
RECONNECT_INTERVAL = float(os.environ.get("RECONNECT_INTERVAL", "30"))
ALERT_COOLDOWN = float(os.environ.get("ALERT_COOLDOWN", "300"))

SERVERCHAN_URL_TEMPLATE = "https://sctapi.ftqq.com/{sendkey}.send"
BEIJING_TZ = timezone(timedelta(hours=8))

ib = IB()
_last_alert_ts = 0.0
_was_connected = False


def now_str() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S %z")


def send_serverchan_alert(title: str, desp: str) -> None:
    """通过 Server酱 (sctapi.ftqq.com) 推送告警到微信——跟改造前完全一样，
    没有改动这部分逻辑。"""
    if not SERVERCHAN_SENDKEY:
        log.warning("SERVERCHAN_SENDKEY 未配置，跳过告警发送。原始消息: %s / %s", title, desp)
        return

    url = SERVERCHAN_URL_TEMPLATE.format(sendkey=SERVERCHAN_SENDKEY)
    payload = {"title": title, "desp": desp}
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        result = json.loads(raw) if raw else {}
        if result.get("code") == 0:
            log.info("已通过 Server酱 发送微信告警")
        else:
            log.error("Server酱 返回异常: %s", result)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        log.error("发送 Server酱 告警失败: %s", e)


def _alert_disconnected(reason: str) -> None:
    """发送"断线"告警，受 ALERT_COOLDOWN 限制，避免持续掉线时反复刷屏。"""
    global _last_alert_ts
    now = time.monotonic()
    if now - _last_alert_ts < ALERT_COOLDOWN:
        log.info("检测到断线（%s），但仍在告警冷却期内（%.0fs），跳过重复告警", reason, ALERT_COOLDOWN)
        return

    title = "ADOS-Trade 告警：IB Gateway 连接断开"
    desp = (
        f"**时间**: {now_str()}\n\n"
        f"**触发方式**: {reason}\n\n"
        "请检查 IB Gateway 容器是否还在运行，必要时通过 VNC（5900 端口）连进容器桌面，"
        "确认账号是否需要重新完成 2FA 确认。"
    )
    send_serverchan_alert(title, desp)
    _last_alert_ts = now


def _on_connected() -> None:
    global _was_connected
    _was_connected = True
    log.info("已连接 IB Gateway（%s:%s，client_id=%s）", HOST, PORT, CLIENT_ID)


def _on_disconnected() -> None:
    """`ib.disconnectedEvent` 回调——断线事件一触发就立刻告警。"""
    global _was_connected
    if _was_connected:
        _was_connected = False
        log.error("disconnectedEvent 触发：连接已断开")
        _alert_disconnected("disconnectedEvent 事件通知")
    else:
        log.info("disconnectedEvent 触发，但已经是断开状态，不重复告警")


ib.connectedEvent += _on_connected
ib.disconnectedEvent += _on_disconnected


def _connect_once() -> bool:
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=10)
        return True
    except Exception as e:  # noqa: BLE001 - 监控脚本需要容错，不能因单次异常退出
        log.error("连接 IB Gateway 失败: %s", e)
        return False


def main() -> None:
    if not SERVERCHAN_SENDKEY:
        log.warning(
            "环境变量 SERVERCHAN_SENDKEY 未设置，一旦 IB Gateway 断线将无法发送告警！"
            "请在 env.list 中配置 SERVERCHAN_SENDKEY。"
        )

    log.info(
        "IB Gateway 连接监控脚本启动，目标：%s:%s，心跳兜底检查间隔：%ss",
        HOST, PORT, HEARTBEAT_INTERVAL,
    )

    while not _connect_once():
        log.info("%.0fs 后重试连接", RECONNECT_INTERVAL)
        ib.sleep(RECONNECT_INTERVAL)

    while True:
        # 用 ib.sleep() 而不是 time.sleep()——ib_insync 靠这个函数在等待期间
        # 继续处理网络事件（包括 disconnectedEvent），用 time.sleep() 会
        # 把事件循环卡死，事件回调也就等不到了。
        ib.sleep(HEARTBEAT_INTERVAL)

        if ib.isConnected():
            log.info("心跳检查：连接正常")
            continue

        # 心跳兜底：走到这里说明 isConnected() already False。如果
        # disconnectedEvent 已经触发过，_was_connected 这时应该已经是
        # False，_alert_disconnected 内部的冷却期会避免重复告警；如果
        # 是 disconnectedEvent 本身漏报的极端情况（_was_connected 仍是
        # True），这里补一次告警。
        log.error("心跳检查发现连接已断开（isConnected()=False）")
        if _was_connected:
            _was_connected = False
            _alert_disconnected("心跳兜底检查发现（disconnectedEvent 未触发）")

        log.info("尝试重新连接...")
        while not _connect_once():
            log.info("%.0fs 后重试连接", RECONNECT_INTERVAL)
            ib.sleep(RECONNECT_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("收到退出信号，监控脚本停止")
        if ib.isConnected():
            ib.disconnect()
        sys.exit(0)
