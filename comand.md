# PhotoPlay 核心命令

## 启动后端
```powershell
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## 重启后端（改了 server/ 代码后，无需重启cpolar）
```powershell
$c = Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -First 1
if ($c) { Stop-Process -Id $c.OwningProcess -Force }
Start-Process python -ArgumentList '-m','uvicorn','server.main:app','--host','0.0.0.0','--port','8000' -WorkingDirectory 'D:\CodeBuddy_ProjectRep\rep\project' -WindowStyle Hidden
```

## 获取公网域名（cpolar 域名一次性，重启cpolar后获取新域名）
```powershell
# 1. 重启 cpolar 服务申请新域名（弹 UAC 确认）
Start-Process powershell -Verb RunAs -ArgumentList '-Command "Restart-Service -Name cpolar -Force"'

# 2. 从日志提取新域名
Get-Content "C:\Users\Jason\.cpolar\logs\cpolar_service.log" -Tail 50 -Encoding UTF8 | Select-String -Pattern "Tunnel established at"
```

## 查看当前公网域名
```powershell
Get-Content "C:\Users\Jason\.cpolar\logs\cpolar_service.log" -Encoding UTF8 | Select-String -Pattern "Tunnel established at" | Select-Object -Last 1
```

## 加入图片/视频后刷新（无需重启后端）
```powershell
# 注册建库：提取特征 + 视频转码优化（faststart，需已安装 ffmpeg，可选）
python tools/register.py
# 热加载特征库与注册表
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/reload
```

## 前端代码修改后
浏览器强刷即可（Ctrl+F5），无需重启后端。
