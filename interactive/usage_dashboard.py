from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .usage_report import build_usage_report


RECENT_EVENT_FIELDS = (
    "event_id",
    "event",
    "time_utc",
    "client_ip",
    "session_id",
    "end_reason",
    "extension_version",
    "previous_version",
    "installation_source",
    "initial_bone_count",
    "final_bone_count",
    "net_bone_change",
    "skin_generation_count",
    "final_has_skin",
    "duration_seconds",
)


def build_usage_dashboard(events: list[dict[str, Any]], *, limit: int = 100) -> dict:
    indexed_events = list(enumerate(events))
    recent = sorted(
        indexed_events,
        key=lambda item: (str(item[1].get("time_utc", "")), item[0]),
        reverse=True,
    )[: max(1, min(int(limit), 500))]
    return {
        "ok": True,
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "event_count": len(events),
        "summary": build_usage_report(events),
        "recent_events": [
            {
                field: event[field]
                for field in RECENT_EVENT_FIELDS
                if field in event
            }
            for _, event in recent
        ],
    }


USAGE_DASHBOARD_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>H3D Skintokens 用量统计</title>
  <style>
    :root {
      color-scheme: light;
      --background: #f4f6f8;
      --surface: #ffffff;
      --border: #dfe3e8;
      --text: #182026;
      --muted: #66717c;
      --accent: #176b4d;
      --accent-soft: #e7f3ed;
      --blue: #245c9c;
      --warning: #9b5d08;
      --danger: #a43d3d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--background);
      color: var(--text);
      font-family: Inter, "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif;
      font-size: 14px;
      letter-spacing: 0;
    }
    button, table { font: inherit; }
    .shell { width: min(1440px, calc(100% - 32px)); margin: 0 auto; padding: 24px 0 40px; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 20px;
    }
    h1 { margin: 0; font-size: 24px; font-weight: 680; line-height: 1.25; }
    .subtitle { margin-top: 5px; color: var(--muted); font-size: 13px; }
    .header-actions { display: flex; align-items: center; gap: 12px; }
    .status { color: var(--muted); white-space: nowrap; }
    .refresh {
      width: 36px;
      height: 36px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface);
      color: var(--text);
      cursor: pointer;
    }
    .refresh:hover { border-color: #aab3bc; background: #f9fafb; }
    .refresh:disabled { cursor: wait; color: #aab3bc; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 22px;
    }
    .metric {
      min-width: 0;
      min-height: 86px;
      padding: 13px 14px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface);
    }
    .metric-label { color: var(--muted); font-size: 12px; line-height: 1.3; }
    .metric-value {
      margin-top: 8px;
      overflow-wrap: anywhere;
      font-size: 24px;
      font-weight: 680;
      line-height: 1;
    }
    .metric[data-tone="green"] .metric-value { color: var(--accent); }
    .metric[data-tone="blue"] .metric-value { color: var(--blue); }
    .metric[data-tone="warning"] .metric-value { color: var(--warning); }
    .section-heading {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 10px;
    }
    h2 { margin: 0; font-size: 16px; font-weight: 680; }
    .event-total { color: var(--muted); font-size: 12px; }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface);
    }
    table { width: 100%; min-width: 860px; border-collapse: collapse; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #edf0f2; text-align: left; vertical-align: top; }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8f9fa;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    tbody tr:last-child td { border-bottom: 0; }
    tbody tr:hover { background: #fafbfb; }
    .event-name {
      display: inline-block;
      padding: 3px 7px;
      border-radius: 4px;
      background: #eef1f4;
      white-space: nowrap;
      font-size: 12px;
      font-weight: 600;
    }
    .event-name[data-event="extension_install"],
    .event-name[data-event="extension_update"] { background: var(--accent-soft); color: var(--accent); }
    .event-name[data-event="session_end"] { background: #eaf0f7; color: var(--blue); }
    .muted, .details { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
    .empty, .error { padding: 44px 16px; text-align: center; color: var(--muted); }
    .error { color: var(--danger); }
    @media (max-width: 1100px) { .metrics { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
    @media (max-width: 640px) {
      .shell { width: min(100% - 20px, 1440px); padding-top: 16px; }
      header { align-items: flex-start; }
      h1 { font-size: 20px; }
      .header-actions { align-items: flex-end; flex-direction: column-reverse; gap: 6px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .metric { min-height: 78px; padding: 11px 12px; }
      .metric-value { font-size: 21px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>H3D Skintokens 用量统计</h1>
        <div class="subtitle">服务使用、插件分发和会话结果</div>
      </div>
      <div class="header-actions">
        <span class="status" id="status">正在加载</span>
        <button class="refresh" id="refresh" type="button" title="刷新" aria-label="刷新">↻</button>
      </div>
    </header>

    <section class="metrics" id="metrics" aria-label="累计统计"></section>

    <section>
      <div class="section-heading">
        <h2>近期事件</h2>
        <span class="event-total" id="event-total"></span>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>时间</th><th>事件</th><th>客户端 IP</th><th>会话</th><th>详情</th></tr></thead>
          <tbody id="events"><tr><td colspan="5" class="empty">正在加载</td></tr></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const metricDefinitions = [
      ["extension_installations", "累计安装设备", "green"],
      ["extension_installs", "插件安装事件", "green"],
      ["extension_updates", "插件更新事件", "green"],
      ["sessions_started", "开始会话", "blue"],
      ["sessions_incomplete", "未结束会话", "warning"],
      ["sessions_finished", "主动完成会话", "blue"],
      ["skinned_assets", "产出蒙皮资产", "green"],
      ["skin_generation_count", "生成 Skin 次数", "blue"],
      ["net_bone_change", "骨骼净增量", ""],
      ["mean_duration", "平均会话时长", ""],
      ["sessions_idle_timed_out", "空闲超时回收", "warning"],
      ["sessions_lru_evicted", "LRU 回收", "warning"]
    ];
    const eventLabels = {
      session_start: "会话开始",
      session_end: "会话结束",
      extension_install: "插件安装",
      extension_update: "插件更新"
    };
    const reasonLabels = {
      finish: "用户完成",
      idle_timeout: "空闲超时",
      lru_evicted: "LRU 回收",
      server_shutdown: "服务关闭",
      replaced: "会话替换"
    };

    function formatNumber(value) {
      return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 1 }).format(Number(value || 0));
    }
    function formatDuration(seconds) {
      const value = Number(seconds || 0);
      if (value < 60) return `${formatNumber(value)} 秒`;
      if (value < 3600) return `${formatNumber(value / 60)} 分钟`;
      return `${formatNumber(value / 3600)} 小时`;
    }
    function formatTime(value) {
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString("zh-CN", { hour12: false });
    }
    function shortId(value) {
      const text = String(value || "");
      return text.length > 12 ? `${text.slice(0, 8)}…` : (text || "-");
    }
    function describe(event) {
      if (event.event === "extension_install") {
        return `版本 ${event.extension_version || "未知"} · ${event.installation_source || "来源未知"}`;
      }
      if (event.event === "extension_update") {
        return `${event.previous_version || "未知"} → ${event.extension_version || "未知"} · ${event.installation_source || "来源未知"}`;
      }
      if (event.event === "session_end") {
        const reason = reasonLabels[event.end_reason] || event.end_reason || "未知原因";
        const bones = Number(event.net_bone_change || 0);
        return `${reason} · ${formatDuration(event.duration_seconds)} · 骨骼 ${bones >= 0 ? "+" : ""}${bones} · Skin ${event.skin_generation_count || 0}`;
      }
      if (event.event === "session_start") {
        return `初始骨骼 ${event.initial_bone_count || 0}`;
      }
      return "-";
    }
    function renderMetrics(summary) {
      const metrics = document.getElementById("metrics");
      metrics.replaceChildren();
      for (const [key, label, tone] of metricDefinitions) {
        const card = document.createElement("div");
        card.className = "metric";
        if (tone) card.dataset.tone = tone;
        const title = document.createElement("div");
        title.className = "metric-label";
        title.textContent = label;
        const value = document.createElement("div");
        value.className = "metric-value";
        value.textContent = key === "mean_duration"
          ? formatDuration(summary.duration_seconds?.mean)
          : formatNumber(summary[key]);
        card.append(title, value);
        metrics.append(card);
      }
    }
    function renderEvents(events) {
      const body = document.getElementById("events");
      body.replaceChildren();
      if (!events.length) {
        const row = body.insertRow();
        const cell = row.insertCell();
        cell.colSpan = 5;
        cell.className = "empty";
        cell.textContent = "暂无事件";
        return;
      }
      for (const event of events) {
        const row = body.insertRow();
        const time = row.insertCell();
        time.textContent = formatTime(event.time_utc);
        time.className = "muted";
        const type = row.insertCell();
        const badge = document.createElement("span");
        badge.className = "event-name";
        badge.dataset.event = event.event || "unknown";
        badge.textContent = eventLabels[event.event] || event.event || "未知事件";
        type.append(badge);
        const ip = row.insertCell();
        ip.textContent = event.client_ip || "-";
        ip.className = "mono";
        const session = row.insertCell();
        session.textContent = shortId(event.session_id);
        session.title = event.session_id || "";
        session.className = "mono";
        const details = row.insertCell();
        details.textContent = describe(event);
        details.className = "details";
      }
    }
    async function refresh() {
      const button = document.getElementById("refresh");
      const status = document.getElementById("status");
      button.disabled = true;
      status.textContent = "正在刷新";
      try {
        const response = await fetch("/v1/usage/summary?limit=100", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        renderMetrics(data.summary || {});
        renderEvents(data.recent_events || []);
        document.getElementById("event-total").textContent = `累计 ${formatNumber(data.event_count)} 条`;
        status.textContent = `更新于 ${formatTime(data.generated_at)}`;
      } catch (error) {
        status.textContent = "加载失败";
        const body = document.getElementById("events");
        body.innerHTML = "";
        const row = body.insertRow();
        const cell = row.insertCell();
        cell.colSpan = 5;
        cell.className = "error";
        cell.textContent = `无法读取统计数据：${error.message}`;
      } finally {
        button.disabled = false;
      }
    }
    document.getElementById("refresh").addEventListener("click", refresh);
    refresh();
    window.setInterval(refresh, 30000);
  </script>
</body>
</html>
"""
