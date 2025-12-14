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

#### 认证与用户管理

- 新增基于 Session 的登录 `/login`, `/logout`, `/me`，客户端通过 HttpOnly 的 Session Cookie 与服务端保持状态。
- 管理用户的 API 统一放在 `/admin/users` 及其相关子路由，管理员必须具有 `role = admin`。
- 密码使用 Argon2id（添加 `argon2-cffi` 依赖）；如果数据库尚无用户，服务会根据 `STOCKTOOL_ADMIN_USERNAME` / `STOCKTOOL_ADMIN_PASSWORD` 自动创建管理员账号（未设置时回退为 `admin`/`admin` 并打印警告，请尽快修改并在首次登录后更新密码）。
- 推荐设置环境变量 `STOCKTOOL_SESSION_SECRET` 来签名 Cookie，并根据部署协议调整 `STOCKTOOL_SESSION_COOKIE_SECURE`（HTTP 下保持 `false`，HTTPS 再设为 `true`）、`STOCKTOOL_SESSION_SAME_SITE`（通常 `lax`）与 `STOCKTOOL_SESSION_MAX_AGE`。启动脚本或 systemd unit 应把这些变量导出，避免默认 secret。
- 确认 `STOCKTOOL_ALLOW_ORIGINS` 包含最终部署的域名/IP（如 `http://xpq.qazwsx123.uk,http://104.207.154.5`）；FastAPI 默认接受 `localhost/127.0.0.1`，你也可以通过 `STOCKTOOL_ALLOW_ORIGIN_REGEX` 精细控制。只有 `Origin` 在白名单、Cookie 设置与请求协议一致时，浏览器才能在 `/me` 返回 200。
- 系统默认关闭登录（`STOCKTOOL_DISABLE_AUTH` 默认为 `1` / `true`，前端 `VITE_DISABLE_AUTH` 默认为非 `false`），所以启动时直接进入仪表盘；如需重新启用认证，仅需把两个变量设为 `0` / `false`。

### 前端（Vue 3 + Vite）

1. 安装依赖并启动开发服务器：
   ```bash
   cd web/frontend
   npm install
   npm run dev -- --host 0.0.0.0
   ```
2. 默认通过 `axios.defaults.baseURL` 调用 `VITE_API_BASE`（未设置时退回到 `/`）。Vite 的 dev server 已配置 `server.proxy` 把 `/api/*` 代理到本地 FastAPI（`http://127.0.0.1:8000`），因此在开发模式下前后端同源、Session Cookie 会自动携带。如果部署在其它域名或网关后面，只需把 `VITE_API_BASE` 设置为完整的后端地址即可；生产打包时可设 `VITE_API_BASE=http://xpq.qazwsx123.uk`。构建后的入口还会在 `index.html` 里自动生成 `window.__STOCKTOOL_RUNTIME_API_BASE`（`http(s)://当前域名:8000`）以便静态部署直接使用当前 host。
   本地调试也可以同时设置 `VITE_DISABLE_AUTH=true`，前端会自动显示主界面，绕过登录流程，与后端的 `STOCKTOOL_DISABLE_AUTH=1` 协同使用即可。
   当前前端已经引入 Vue Router 构建多页面体验：`/login` 展示登录表单，`/register` 提供自助注册，登录后 `/` 展示回测主界面，管理员用户可在顶部点击跳转 `/admin` 管理账号。Axios 默认会设置 `withCredentials = true`，所有认证信息由 HttpOnly Cookie 保存，不要在客户端存储令牌。

   管理后台现在可以在用户列表中直接修改角色、下发密码重置、永久禁用/启用、设置临时禁用以及删除用户。对应的 API 包括 `PUT /admin/users/{id}`、`POST /admin/users/{id}/reset-password`、`DELETE /admin/users/{id}`，并仍然遵循“不能禁用/删除自己”与“至少保留一个管理员” 的保护逻辑。
   安装依赖时请确保包含新依赖 `vue-router`（已列入 `package.json`），任何新增依赖都需要重新执行 `npm install`。
3. 生产环境构建：
   ```bash
   VITE_API_BASE=http://xpq.qazwsx123.uk npm run build
   ```
   构建产物位于 `web/frontend/dist/`，将 `dist/` 内容复制到 nginx 静态根目录（例如 `/var/www/stocktool`），同时确保 nginx 或代理把 `/api/` 请求转发到 `http://127.0.0.1:8000`（保留 `stocktool_session` cookie）。

## 部署建议

**后端**：使用 systemd 或其他进程管理器运行 `uvicorn web.backend.main:app --host 0.0.0.0 --port 8000`，并在 service 定义或启动脚本里导出关键环境变量：
  ```bash
  export STOCKTOOL_SESSION_SECRET="你自己的随机串"
  export STOCKTOOL_ALLOW_ORIGINS="http://104.207.154.5,http://xpq.qazwsx123.uk"
  export STOCKTOOL_SESSION_COOKIE_SECURE=false   # HTTP 时设为 false
  export STOCKTOOL_SESSION_SAME_SITE=lax
  export STOCKTOOL_ADMIN_PASSWORD="你想要的强密码"
  ```
  设置完成后重启服务（`sudo systemctl daemon-reload && sudo systemctl restart stocktool`），确保 `/me` 能正确读取 session。
**前端**：将 `dist/` 部署到任意静态站点（Nginx、Cloudflare Pages、GitHub Pages 等），并将 `VITE_API_BASE` 指向部署域名（例如 `VITE_API_BASE=http://xpq.qazwsx123.uk`）。构建后的入口会通过 `window.__STOCKTOOL_RUNTIME_API_BASE` 默认指向当前 host 的 `http(s)://<host>:8000`，方便直接托管在 nginx 上。
**Nginx 示例**：
  ```nginx
  server {
      listen 80;
      listen [::]:80;
      server_name xpq.qazwsx123.uk 104.207.154.5;
      root /var/www/stocktool;
      index index.html;
      location / {
          try_files $uri $uri/ /index.html;
      }
      location /api/ {
          proxy_pass http://127.0.0.1:8000/;
          proxy_http_version 1.1;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
          proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
          proxy_set_header X-Forwarded-Proto $scheme;
          proxy_set_header Connection "";
          proxy_redirect off;
          proxy_cookie_path / /;
      }
  }
  ```

## 常见问题

- **上传失败 / Network Error**：确认前端的 `VITE_API_BASE` 指向生产 API（不要留 `127.0.0.1`），并用浏览器 DevTools 检查 `/login` 是否返回 `Set-Cookie: stocktool_session`、后续 `/me` 请求是否带上 Cookie 并返回 200；若缺失，检查 nginx 是否把 `/api/`（前端 proxy）正确代理至 `http://127.0.0.1:8000`，以及 `STOCKTOOL_ALLOW_ORIGINS` 与 `STOCKTOOL_SESSION_COOKIE_SECURE` 是否与部署协议匹配（HTTP 时设为 `false`）。
- **500 PermissionError**：给 `web/backend/uploads/` 与 `results/` 目录赋予运行用户写权限。
- **Git dubiously-owned repository**：在服务器上执行 `git config --global --add safe.directory /opt/StockTool` 解决。
- **用户数据库 users.db 被覆盖**：`web/backend/users.db` 已列入 `.gitignore`，默认不会被 `git pull`/`git push` 覆盖；如果怕出错，只要不删掉这个文件，它会保留你当前的 admin/用户记录。

欢迎根据业务需求扩展更多策略或可视化模块，贡献 PR！
