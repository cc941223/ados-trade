# ADOS-Trade

个人交易决策仪表盘，基于 IBKR 真实数据，覆盖日内短线 / 波段趋势 / 期权策略 / 杠杆 ETF 对冲四大场景。

详细数据架构与开发计划见 `docs/ADOS-Trade_数据架构与技术规格书_v1.0.docx`。

## 当前阶段：M0 —— 基础设施与可行性验证

### 待办
- [ ] 部署 IBeam + TimescaleDB + Redis（本仓库已提供 docker-compose.yml）
- [ ] 首次登录完成 IBKR 2FA 确认
- [ ] 验证 Gateway 是否正常返回账户数据
- [ ] 将 Gateway 登出策略改为 Auto-Restart
- [ ] 直连 CPAPI 复测：分钟线历史可回溯天数
- [ ] 直连 CPAPI 复测：Greeks 字段（7308 Delta / 7309 Gamma / 7310 Theta / 7311 Vega）

### 快速开始
1. 复制 `env.list.example` 为 `env.list`，填入 IBKR 账号密码
2. `docker compose up -d`
3. 手机 IB Key 推送确认登录
4. 验证：`curl https://localhost:5000/v1/api/portfolio/accounts -k`

## 目录结构
```
ados-trade/
├── docker-compose.yml   # IBeam + TimescaleDB + Redis
├── env.list.example     # 环境变量模板
├── engine/              # ADOS Engine 指标计算层（M2 阶段开始填充）
├── data/                # 数据采集脚本（M1 阶段开始填充）
└── docs/                # 项目文档
```
