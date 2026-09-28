"""ui.components — 组件导出

  indicators  版式 / KPI / 徽章 / 卡片
  charts      Plotly + ECharts 图表构建器
"""
from ui.components.indicators import (  # noqa: F401
    algorithm_badge, advice_card, audit_badge, callout, dev_table, empty_state,
    kpi_card, kpi_row, legend_html, mode_badge, section_header, severity_tone,
    status_chip,
)
from ui.components.charts import (  # noqa: F401
    donut, heatmap, hbar_simple, matrix_heatmap, quadrant_chart, radar, rank_bars,
    sankey, sentiment_bars_by_attr, sentiment_hist, sr_scatter, sunburst, trend_line,
)
