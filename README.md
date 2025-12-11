# StockTool

StockTool 是一个以 Python 为核心的量化回测与可视化套件，包含 FastAPI 提供的数据/回测服务与 Vue/ECharts 打造的前端仪表盘。支持上传 CSV 行情、编写通达信公式并一键运行多种策略回测，方便在本地或 VPS 上向团队演示结果。

## 功能概览

- 上传任意包含 `date/open/high/low/close` 的行情 CSV，并通过自定义公式生成买卖信号。
- 直接在界面输入股票代码，自动从新浪行情接口抓取最新历史数据，无需手动准备 CSV。
- 内置多种策略：固定持有周期、止盈止损、定投、简单网格等，可组合回测并导出结果。
- 提供盈亏曲线、交易列表、指标评分、压力测试、仓位计划等多维分析。
- 前端支持一键触发回测、查看日志、展示多周期信号与复盘摘要。

## 目录结构

```
StockTool/
├─ analytics.py           # 各类指标与分析工具
├─ backtest_service.py    # 组织回测流程并输出结果
├─ backtesting.py         # 具体策略实现
├─ formula_engine.py      # 通达信公式解析执行
├─ web/
│  ├─ backend/            # FastAPI 应用 (main.py)
│  └─ frontend/           # Vue + Vite 前端
└─ ...其它脚本/配置
```

## 本地开发

### 后端（FastAPI）

1. 创建并激活虚拟环境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 安装依赖（根据需要补充 ta-lib、matplotlib 等）：
   ```bash
   pip install fastapi uvicorn pandas numpy python-multipart requests
   ```
3. 启动服务：
   ```bash
   uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. 上传接口将 CSV 保存到 `web/backend/uploads/`，回测结果写入 `results/`，确保这些目录可写。

### 前端（Vue 3 + Vite）

1. 安装依赖并启动开发服务器：
   ```bash
   cd web/frontend
   npm install
   npm run dev -- --host 0.0.0.0
   ```
2. 默认通过 `axios.defaults.baseURL` 调用 `VITE_API_BASE`。本地可以在 `web/frontend/.env.development` 中设置：
   ```env
   VITE_API_BASE=http://127.0.0.1:8000
   ```
3. 生产环境构建：
   ```bash
   npm run build
   ```
   构建产物位于 `web/frontend/dist/`。

## 部署建议

- **后端**：使用 systemd 或其他进程管理器运行 `uvicorn web.backend.main:app --host 0.0.0.0 --port 8000`，开放端口或通过 Nginx 反向代理至域名（支持 HTTPS）。
- **前端**：将 `dist/` 部署到任意静态站点（Nginx、Cloudflare Pages、GitHub Pages 等）。在 `.env.production` 中配置 API 地址，例如：
  ```env
  VITE_API_BASE=https://api.example.com/api
  ```
- **Nginx 示例**：
  ```nginx
  server {
      listen 80;
      server_name example.com;
      root /var/www/stocktool;
      index index.html;
      location / { try_files $uri $uri/ /index.html; }
      location /api/ {
          proxy_pass http://127.0.0.1:8000/;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
      }
  }
  ```

## 常见问题

- **上传失败 / Network Error**：确认前端的 `VITE_API_BASE` 指向真实 API 地址，不要保留 `127.0.0.1`。
- **500 PermissionError**：给 `web/backend/uploads/` 与 `results/` 目录赋予运行用户写权限。
- **Git dubiously-owned repository**：在服务器上执行 `git config --global --add safe.directory /opt/StockTool` 解决。

欢迎根据业务需求扩展更多策略或可视化模块，贡献 PR！
