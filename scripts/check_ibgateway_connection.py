#!/usr/bin/env python3
"""手动连通性检查脚本 —— 不是 pytest 单元测试，需要人手动跑，连一个真实
在跑的 IB Gateway（`docker compose up -d ibgateway` 之后），确认：

1. ib_insync 能连上 TWS API（默认 4002 端口）。
2. 连上之后能拿到账户信息（`managedAccounts`/`accountSummary`）。
3. `brokers.ibkr.client.IBInsyncClient.get_stock_ohlcv` 能拿到一小段真实
   历史数据（顺手验证这条链路，不只是"连上了"）。

用法：
    python3 scripts/check_ibgateway_connection.py
    # 或者用环境变量覆盖默认连接参数：
    IBGATEWAY_HOST=127.0.0.1 IBGATEWAY_PORT=4002 IBGATEWAY_CLIENT_ID=99 \
        python3 scripts/check_ibgateway_connection.py

环境变量（都可选，不设就用括号里的默认值）：
    IBGATEWAY_HOST         Gateway 地址（127.0.0.1）
    IBGATEWAY_PORT         TWS API 端口（4002，跟 docker-compose.yml 里
                           `ibgateway` 服务映射的端口一致；4002 是 paper
                           账户默认端口）
    IBGATEWAY_CLIENT_ID    TWS API client id（99，故意跟 `IBInsyncClient`
                           默认值 1 不一样，避免这个手动检查脚本和真正
                           跑数据采集的客户端抢同一个 client id 把对方
                           踢掉）
    CHECK_SYMBOL           用来试探 get_stock_ohlcv 的标的，默认 "AAPL"

退出码：0 = 全部检查通过；1 = 连接或任一检查失败。
"""
from __future__ import annotations

import os
import sys

from ib_insync import IB

# 让脚本可以直接用 `python3 scripts/check_ibgateway_connection.py` 跑，
# 不需要先 `pip install -e .` 或者设置 PYTHONPATH。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brokers.ibkr.client import IBInsyncClient  # noqa: E402  (需要先插入 sys.path)


def main() -> int:
    host = os.environ.get("IBGATEWAY_HOST", "127.0.0.1")
    port = int(os.environ.get("IBGATEWAY_PORT", "4002"))
    client_id = int(os.environ.get("IBGATEWAY_CLIENT_ID", "99"))
    check_symbol = os.environ.get("CHECK_SYMBOL", "AAPL")

    print(f"[1/3] 连接 IB Gateway：{host}:{port}（client_id={client_id}）...")
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=10)
    except Exception as exc:  # noqa: BLE001 - 手动检查脚本，任何连接异常都要清楚打印出来
        print(f"❌ 连接失败：{exc}")
        print("   检查清单：")
        print("   - ibgateway 容器是否已经跑起来（docker compose ps）？")
        print("   - 首次启动是否已经通过 VNC（5900 端口）完成登录/2FA 确认？")
        print("   - IBGATEWAY_PORT 是否跟 docker-compose.yml 里映射的端口一致？")
        return 1

    try:
        print("✅ 连接成功")

        print(f"\n[2/3] 拉取账户信息...")
        accounts = ib.managedAccounts()
        if not accounts:
            print("❌ managedAccounts() 返回空列表——连接建立了，但没有可用账户")
            return 1
        print(f"✅ 可用账户：{accounts}")

        summary = ib.accountSummary(accounts[0])
        if summary:
            print(f"   账户 {accounts[0]} 关键字段：")
            interesting_tags = {"NetLiquidation", "TotalCashValue", "BuyingPower", "AvailableFunds"}
            for row in summary:
                if row.tag in interesting_tags:
                    print(f"     {row.tag}: {row.value} {row.currency}")
        else:
            print("⚠️  accountSummary() 返回空——连接和账户都正常，可能只是这个账户暂时没有可展示的字段")

        print(f"\n[3/3] 通过 IBInsyncClient.get_stock_ohlcv 验证一条真实数据链路（symbol={check_symbol}）...")
        client = IBInsyncClient(host=host, port=port, client_id=client_id + 1)
        rows = client.get_stock_ohlcv(check_symbol, "1day", start=None, end=None)
        if not rows:
            print(f"❌ get_stock_ohlcv 返回空列表——连接正常，但没拿到 {check_symbol} 的历史数据")
            return 1
        print(f"✅ 拿到 {len(rows)} 根日线，最近一根：{rows[-1]}")
        client.disconnect()

    finally:
        ib.disconnect()

    print("\n全部检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
