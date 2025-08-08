
虚拟机资源方面，单个账号可以拥有：
- 2 个基于 AMD 的虚拟机，每个虚拟机配备 1/8 OCPU 和 1 GB 内存
- 基于 Arm 的 Ampere A1 内核和 24 GB 内存，可作为 1 个虚拟机或最多 4 个虚拟机使用，每月有 3000 个 OCPU 小时和 18000 GB 小时。（每个月 24*30 = 720 小时，**意味着你的 4C24G 的 ARM 服务器可以全天候免费在线**）
- 存储方面，有 200G 的免费块存储

为了实现最大化利用免费实例
- 单台 ARM 4C/24GB/200G（性能存储带宽最大化）
- 单台 ARM 4C/24GB/100G + 两台 X86 1C/1G/50G（性能和数量均衡选择）


## 如何开放oracle的实例端口

要让你的 Oracle Cloud 实例上的 Web 应用（在端口 5173 运行）能从外部网络访问，你需要做两件事：

### 1. 开放 Oracle Cloud 网络安全列表的端口

Oracle Cloud 默认会限制进出的网络流量。你需要去 Oracle Cloud 控制台，找到你的虚拟云网络（VCN），然后修改相关的**安全列表 (Security List)**。

找到你的实例所使用的安全列表，然后添加一个入站规则（Ingress Rule）：

- **源 CIDR (Source CIDR)**：`0.0.0.0/0` (这表示允许来自任何 IP 地址的流量)
    
- **IP 协议 (IP Protocol)**：`TCP`
    
- **源端口范围 (Source Port Range)**：全部（留空即可）
    
- **目标端口范围 (Destination Port Range)**：`5173`
    

这个设置告诉 Oracle Cloud 的防火墙，允许任何 IP 地址通过 TCP 协议访问你的实例的 5173 端口。

### 2. 开放实例操作系统的防火墙端口

即使 Oracle Cloud 的网络防火墙放行了流量，你的实例操作系统内部的防火墙（比如 Linux 系统上的 `firewalld` 或 `iptables`）也可能阻止访问。

你可以使用 UFW （ubuntu）来开放 5173 端口。

1. **检查 UFW 状态** 首先，查看 UFW 是否已经启用。
    ```
    sudo ufw status
    ```
    如果显示 `Status: inactive`，你需要先启用它。
    
2. **启用 UFW 并开放端口** 如果 UFW 处于非活动状态，你可以按照以下步骤操作：

    ```
    # 启用 UFW
    sudo ufw enable
    
    # 开放 5173 端口（允许 TCP 协议）
    sudo ufw allow 5173/tcp
    
    # 再次检查状态，确认规则已生效
    sudo ufw status
    ```
    
    `ufw status` 的输出应该会显示一条规则，类似这样：
    
    Plaintext
    
    ```
    Status: active
    
    To                         Action      From
    --                         ------      ----
    5173/tcp                   ALLOW       Anywhere
    ```
    

---

完成这些步骤后，你的 Ubuntu 实例应该已经开放了 5173 端口。结合之前在 Oracle Cloud 控制台安全列表中的设置，你应该就可以从外部网络访问你的 Web 应用了。

地址
```
http://129.146.33.58:5173
```

### 检查端口

使用 ss 命令
```
ss -antp | grep 5173

OR

netstat -antp | grep 5173
```

如果输出显示的是 `127.0.0.1:5173`，那么你的应用就只在本地监听

#### vite react application

你需要编辑你的项目根目录下的 `vite.config.js`（或 `vite.config.ts`）文件。
在 `defineConfig` 函数中，添加或修改 `server` 选项，并设置 `host: true`。

```
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // 将此行添加或修改
    port: 5173, // 你应用的端口
  }
})
```