# Claude Code CLI 本地安装计划

## 目标

在当前用户的 Node.js 环境中安装 Claude Code CLI，并确认 `claude` 命令可执行。

## 现状

- Node.js：`v22.12.0`
- npm：`10.9.0`
- npm 全局前缀：`/home/rpartx3080/.nvm/versions/node/v22.12.0`
- 当前未发现可执行的 `claude` 命令。
- `@anthropic-ai/claude-code` npm `latest`：`2.1.258`

## 决策

- 使用当前 nvm Node 环境执行 `npm install -g @anthropic-ai/claude-code`，安装范围为当前用户的 Node 全局环境，不修改项目依赖。
- 安装后执行 `claude --version` 与命令路径检查。
- 认证不在本次安装范围内；首次运行 Claude Code 时按 CLI 提示登录即可。

## 验证标准

- npm 全局安装成功。
- `claude` 能被 PATH 找到。
- `claude --version` 返回已安装版本。

