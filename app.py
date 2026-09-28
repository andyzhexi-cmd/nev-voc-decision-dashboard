"""
智评车行 · Streamlit 看板主入口
=====================================================================
新能源汽车用户评论分析与产品改进决策平台
深色玻璃拟态风格 · 全宽顶部 Tab · Plotly 交互图表
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "data" / "raw" / "comments_raw.csv"
CLEAN = ROOT / "data" / "processed" / "comments_cleaned.csv"
CLUSTERED = ROOT / "data" / "processed" / "comments_clustered.csv"
SENTIMENT = ROOT / "data" / "processed" / "sentiment_results.csv"
VIKOR = ROOT / "outputs" / "results" / "vikor_results.csv"
ISA = ROOT / "outputs" / "results" / "isa_results.csv"
FIG_ELBOW = ROOT / "outputs" / "figures" / "01_elbow_silhouette.png"
FIG_CLUSTER = ROOT / "outputs" / "figures" / "02_cluster_distribution.png"

st.set_page_config(page_title="智评车行", layout="wide", page_icon="🚗", initial_sidebar_state="collapsed")

# ============ 全局样式：深色玻璃拟态 ============
st.markdown("""
<style>
/* 全局深色背景 */
.stApp {
    background: #0a0a0f;
    color: #e8e8ed;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "PingFang SC", sans-serif;
}

/* 主内容区去掉 padding，tab 顶到头 */
.main .block-container {
    padding-top: 0;
    padding-left: 0;
    padding-right: 0;
    max-width: 100%;
}

/* ============ 顶部 Tab 栏（全宽顶到头）============ */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    border-radius: 0;
    padding: 0 24px;
    gap: 0;
    position: sticky;
    top: 0;
    z-index: 999;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
}

[data-baseweb="tab"] {
    border-radius: 0;
    padding: 16px 24px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: rgba(255,255,255,0.6);
    font-weight: 500;
    font-size: 15px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.9);
}

[data-baseweb="tab"][aria-selected="true"] {
    background: transparent;
    color: #ffffff;
    border-bottom: 2px solid #6366f1;
    box-shadow: none;
}

[data-baseweb="tab-border"] { display: none; }

/* Tab 内容区 padding */
.stTabs [data-baseweb="tab-panel"] {
    padding: 32px 40px;
}

/* ============ 标题 ============ */
h1, h2, h3 {
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}

/* 标题区渐变效果 */
h1 {
    background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ============ Metric 卡片（玻璃拟态）============ */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

[data-testid="stMetric"]:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(99,102,241,0.3);
    transform: translateY(-2px);
}

[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.5) !important;
    font-size: 13px !important;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

/* ============ 文本 ============ */
p, span, div {
    color: rgba(255,255,255,0.85);
}

/* ============ 分隔线 ============ */
hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 32px 0 !important;
}

/* ============ 表格 ============ */
.stDataFrame {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    overflow: hidden;
}

/* ============ Info/Success/Warning 卡片 ============ */
.stAlert {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.9) !important;
}

/* ============ Plotly 图表容器 ============ */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* ============ 隐藏 Streamlit 顶部菜单 ============ */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ============ 滚动条 ============ */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }
</style>
""", unsafe_allow_html=True)

# 顶部标题区
st.markdown("""
<div style="padding: 40px 40px 20px 40px;">
    <h1 style="margin: 0; font-size: 36px;">智评车行</h1>
    <p style="color: rgba(255,255,255,0.5); margin: 8px 0 0 0; font-size: 15px;">新能源汽车用户评论分析与产品改进决策平台</p>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if path.exists() else None


# ============ 五个顶部 Tab ============
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 数据概览", "🏷️ 属性聚类", "💬 情感分析", "🎯 决策排序", "📐 ISA矩阵"]
)

# ============================================================
# Tab 1: 数据概览
# ============================================================
with tab1:
    df = load_csv(RAW)
    if df is None:
        st.warning("尚未生成数据，请先运行：./vene/bin/python -m src.data_ingest.mock_generator")
    else:
        n_total = len(df)
        n_noise = int((df["_attr_ground_truth"] == "噪声").sum())
        valid = n_total - n_noise

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("评论总数", f"{n_total:,}")
        c2.metric("平台数", df["platform"].nunique())
        c3.metric("品牌数", df["brand"].nunique())
        c4.metric("有效评论", f"{valid:,}")
        c5.metric("噪声评论", f"{n_noise:,}")

        st.divider()
        left, right = st.columns(2)
        with left:
            st.subheader("各平台评论量")
            plat_counts = df.groupby("platform").size().sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                x=plat_counts.values, y=plat_counts.index, orientation='h',
                marker=dict(color='#6366f1', opacity=0.8),
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='rgba(255,255,255,0.6)'),
                yaxis=dict(color='rgba(255,255,255,0.9)'),
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("各品牌评论量")
            brand_counts = df.groupby("brand").size().sort_values(ascending=True)
            fig2 = go.Figure(go.Bar(
                x=brand_counts.values, y=brand_counts.index, orientation='h',
                marker=dict(color='#8b5cf6', opacity=0.8),
            ))
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='rgba(255,255,255,0.6)'),
                yaxis=dict(color='rgba(255,255,255,0.9)'),
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        df_clean = load_csv(CLEAN)
        if df_clean is not None:
            st.divider()
            st.subheader("预处理效果")
            dropped = n_total - len(df_clean)
            c1, c2, c3 = st.columns(3)
            c1.metric("原始评论", f"{n_total:,}")
            c2.metric("清洗后", f"{len(df_clean):,}")
            c3.metric("剔除", f"{dropped:,} ({dropped/n_total*100:.1f}%)")

        st.divider()
        st.subheader("评论样例（随机 10 条）")
        st.dataframe(
            df.sample(10, random_state=42)[
                ["platform", "brand", "model", "comment_date", "comment_text"]
            ],
            use_container_width=True, hide_index=True,
        )

# ============================================================
# Tab 2: 属性聚类
# ============================================================
with tab2:
    st.subheader("TF-IDF + K-means 六大属性识别")

    if FIG_ELBOW.exists():
        st.image(str(FIG_ELBOW), caption="肘部法则 + 轮廓系数（论文图5.6）", use_container_width=True)

    df_c = load_csv(CLUSTERED)
    if df_c is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("聚类纯度", "0.502", help="论文基准 0.853（真实数据）")
        with col2:
            st.metric("K=6 轮廓系数", "0.106", help="论文基准 0.598（真实数据）")

        st.caption("注：仿真数据模板化导致聚类纯度低于论文；真实数据（八爪鱼采集）将自动启用无监督聚类。")

    if FIG_CLUSTER.exists():
        st.image(str(FIG_CLUSTER), caption="簇分布（论文图5.7）", use_container_width=True)

    st.divider()
    st.subheader("六大属性")
    attr_info = pd.DataFrame({
        "属性": ["外观", "内饰", "空间", "续航", "性价比", "舒适性"],
        "核心词": [
            "外观/耐看/车身/颜色/造型/大灯/轮毂",
            "内饰/方向盘/屏幕/座椅/真皮/音响/空调",
            "宽敞/空间/紧凑/前排/后排/后备箱/腿部",
            "续航/电耗/充电/电池/公里/能耗/充电桩",
            "价格/便宜/实用/家用/耐用/省钱/满意",
            "舒服/隔音/噪音/加热/调节/科技/平稳",
        ],
    })
    st.dataframe(attr_info, use_container_width=True, hide_index=True)

# ============================================================
# Tab 3: 情感分析
# ============================================================
with tab3:
    st.subheader("双模型情感分析（VADER + 朴素贝叶斯）")

    df_s = load_csv(SENTIMENT)
    if df_s is not None:
        kept = df_s[df_s["dual_agree"] == True]
        c1, c2, c3 = st.columns(3)
        c1.metric("总评论", f"{len(df_s):,}")
        c2.metric("双模型一致（保留）", f"{len(kept):,} ({len(kept)/len(df_s)*100:.1f}%)")
        c3.metric("不一致（剔除）", f"{len(df_s)-len(kept):,}")

        st.divider()
        st.subheader("各属性平均情感值")
        attr_sent = kept.groupby("_attr_ground_truth")["final_sentiment"].agg(["mean", "count"])
        attr_sent.columns = ["平均情感值", "评论数"]
        attr_sent = attr_sent.reindex(["外观", "内饰", "空间", "续航", "性价比", "舒适性"])

        # Plotly 横向条形图
        fig_sent = go.Figure(go.Bar(
            x=attr_sent["平均情感值"].values,
            y=attr_sent.index,
            orientation='h',
            marker=dict(
                color=attr_sent["平均情感值"].values,
                colorscale='RdYlGn',
                opacity=0.85,
            ),
            text=[f"{v:.3f}" for v in attr_sent["平均情感值"].values],
            textposition='auto',
        ))
        fig_sent.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='rgba(255,255,255,0.6)'),
            yaxis=dict(color='rgba(255,255,255,0.9)'),
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_sent, use_container_width=True)

        st.dataframe(attr_sent, use_container_width=True)

        st.divider()
        st.subheader("模型性能对比")
        perf = pd.DataFrame({
            "模型": ["VADER", "朴素贝叶斯", "双模型融合", "论文-双模型"],
            "Precision": [0.987, 0.999, 1.000, 0.906],
            "Recall": [0.877, 1.000, 1.000, 0.933],
            "F1": [0.929, 0.999, 1.000, 0.919],
        })
        st.dataframe(perf, use_container_width=True, hide_index=True)
        st.caption("注：仿真数据模板规整，指标偏高；真实数据将接近论文基准。")
    else:
        st.warning("请先运行阶段4：./vene/bin/python -m src.sentiment.run_sentiment")

# ============================================================
# Tab 4: 决策排序
# ============================================================
with tab4:
    st.subheader("AHP + DEMATEL + PLTS-VIKOR 属性重要性排序")

    col1, col2, col3 = st.columns(3)
    col1.metric("AHP 权重 wA", "(0.434, 0.187, 0.379)")
    col2.metric("DEMATEL 权重 wD", "(0.516, 0.204, 0.280)")
    col3.metric("综合权重 w (λ=0.5)", "(0.475, 0.196, 0.329)")

    st.divider()
    st.subheader("属性重要性 P(x_i) 排序")
    df_v = load_csv(VIKOR)
    if df_v is not None:
        df_v_sorted = df_v.sort_values("Pxi", ascending=True)

        fig_v = go.Figure(go.Bar(
            x=df_v_sorted["Pxi"].values,
            y=df_v_sorted["属性"].values,
            orientation='h',
            marker=dict(
                color=df_v_sorted["Pxi"].values,
                colorscale='Blues',
                opacity=0.85,
            ),
            text=[f"{v:.2f}" for v in df_v_sorted["Pxi"].values],
            textposition='auto',
        ))
        fig_v.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='rgba(255,255,255,0.6)'),
            yaxis=dict(color='rgba(255,255,255,0.9)'),
            height=320, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_v, use_container_width=True)

        df_v_disp = df_v.sort_values("Pxi", ascending=False).reset_index(drop=True)
        df_v_disp.index = df_v_disp.index + 1
        st.dataframe(df_v_disp, use_container_width=True)
    else:
        st.warning("请先运行阶段5：./vene/bin/python -m src.decision.run_decision")

    st.divider()
    st.subheader("排序结论")
    st.success("**性价比** 重要性最高，**续航** 次之，**外观** 最低")
    st.caption("论文基准：性价比(3.4675) > 续航(2.6432) > 空间(1.9752) > 内饰(1.7604) > 舒适性(1.4639) > 外观(1.4073)")

# ============================================================
# Tab 5: ISA矩阵
# ============================================================
with tab5:
    st.subheader("重要性-满意度（ISA）四象限矩阵")

    df_i = load_csv(ISA)
    if df_i is not None:
        quad_colors = {
            "保持区": "#22c55e",
            "机会区": "#3b82f6",
            "低优先级区": "#6b7280",
            "改进区": "#ef4444",
        }

        fig = go.Figure()

        # 象限背景
        fig.add_hrect(y0=3.43, y1=4.5, x0=0.5, x1=1, fillcolor="#22c55e", opacity=0.06, layer="below")
        fig.add_hrect(y0=2.5, y1=3.43, x0=0.5, x1=1, fillcolor="#ef4444", opacity=0.06, layer="below")

        # 分隔线
        fig.add_vline(x=3.55, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)
        fig.add_hline(y=3.43, line_dash="dash", line_color="rgba(255,255,255,0.3)", line_width=1)

        # 散点
        for _, row in df_i.iterrows():
            color = quad_colors.get(row["象限"], "#6b7280")
            fig.add_trace(go.Scatter(
                x=[row["重要性"]],
                y=[row["满意度"]],
                mode="markers+text",
                marker=dict(size=20, color=color, line=dict(color="white", width=2.5)),
                text=[row["属性"]],
                textposition="top center",
                textfont=dict(size=13, color="#ffffff"),
                hovertemplate=(
                    f"<b>{row['属性']}</b><br>"
                    f"重要性: {row['重要性']:.1f}<br>"
                    f"满意度: {row['满意度']:.1f}<br>"
                    f"象限: {row['象限']}<extra></extra>"
                ),
                showlegend=False,
            ))

        # 象限标签
        fig.add_annotation(x=3.0, y=4.3, text="机会区", font=dict(size=14, color="rgba(255,255,255,0.4)"), showarrow=False)
        fig.add_annotation(x=4.05, y=4.3, text="保持区", font=dict(size=14, color="rgba(255,255,255,0.4)"), showarrow=False)
        fig.add_annotation(x=3.0, y=2.62, text="低优先级区", font=dict(size=14, color="rgba(255,255,255,0.4)"), showarrow=False)
        fig.add_annotation(x=4.05, y=2.62, text="改进区", font=dict(size=14, color="rgba(255,255,255,0.4)"), showarrow=False)

        fig.update_layout(
            xaxis_title="重要性（Importance）",
            yaxis_title="满意度（Satisfaction）",
            xaxis=dict(range=[2.5, 4.5], gridcolor="rgba(255,255,255,0.08)", color="rgba(255,255,255,0.6)"),
            yaxis=dict(range=[2.5, 4.5], gridcolor="rgba(255,255,255,0.08)", color="rgba(255,255,255,0.6)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=550,
            margin=dict(l=60, r=40, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.dataframe(df_i[["属性", "重要性", "满意度", "象限"]], use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("象限解读")
        c1, c2 = st.columns(2)
        with c1:
            st.info("**保持区**：性价比\n用户很看重，满意度也高——继续保持技术领先")
            st.warning("**改进区**：续航\n用户很看重，但满意度低——重点改进方向")
        with c2:
            st.success("**机会区**：空间\n用户不太看重，但满意度高——差异化亮点")
            st.caption("**低优先级区**：外观/内饰/舒适性\n用户不太看重，满意度也一般——投入资源回报低")
