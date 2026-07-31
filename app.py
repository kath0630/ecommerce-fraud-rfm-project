from pathlib import Path
from datetime import datetime
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR
MODEL_PATH = APP_DIR / "histgradientboosting.joblib"
THRESHOLD = 0.9344542782645281

CATEGORY_ZH = {
    "entertainment": "娛樂", "food_dining": "餐飲", "gas_transport": "加油／交通",
    "grocery_net": "線上雜貨", "grocery_pos": "實體雜貨", "health_fitness": "健康／健身",
    "home": "居家", "kids_pets": "兒童／寵物", "misc_net": "線上其他",
    "misc_pos": "實體其他", "personal_care": "個人照護", "shopping_net": "線上購物",
    "shopping_pos": "實體購物", "travel": "旅遊",
}
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
]

CLUSTER_INFO = pd.DataFrame([
    ["客群 0", "低活躍／待喚回", 222, "交易頻率較低、距最近交易較久", "再行銷、回購優惠"],
    ["客群 1", "中頻夜間活躍", 334, "夜間交易比例較高、消費頻率中等", "夜間推播、情境式優惠"],
    ["客群 2", "高頻穩定", 182, "交易頻率最高、平均金額較低", "會員維繫、交叉銷售"],
    ["客群 3", "高價值", 170, "總消費與平均金額較高", "VIP 權益、專屬服務"],
], columns=["群組", "中文命名", "卡片數", "主要特徵", "行銷建議"])


st.set_page_config(
    page_title="交易詐欺風險辨識與顧客分群",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {background: #f7f9fc;}
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {
        background: white; border: 1px solid #e3e9f2; border-radius: 14px;
        padding: 14px 18px; box-shadow: 0 5px 16px rgba(18, 54, 95, .05);
    }
    .hero {
        padding: 28px 32px; border-radius: 20px; color: white;
        background: linear-gradient(120deg, #073b6f, #1261a0 65%, #168aad);
        margin-bottom: 22px;
    }
    .hero h1 {margin: 0 0 6px; font-size: 2.15rem;}
    .hero p {margin: 0; opacity: .92;}
    .result-high, .result-low {
        padding: 20px; border-radius: 15px; font-size: 1.05rem; margin: 12px 0;
    }
    .result-high {background:#fff0f0; border-left:6px solid #d64545;}
    .result-low {background:#eaf8f0; border-left:6px solid #278a58;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def signed_log(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def make_features(
    amount, age, distance, city_pop, tx_date, hour, category, gender, state,
    prior_count, previous_amount, minutes_since_previous, prior_mean,
    merchant_frequency, job_frequency, city_frequency, category_frequency,
    state_frequency,
):
    prior = prior_count > 0
    seconds = max(minutes_since_previous * 60, 0) if prior else 0
    previous = previous_amount if prior else 0
    prior_avg = prior_mean if prior else 0
    ratio = amount / max(prior_avg, 1.0) if prior else 1.0
    ratio = float(np.clip(ratio, 0.01, 100.0))
    return pd.DataFrame([{
        "amt_log1p": np.log1p(max(amount, 0)),
        "age": age,
        "distance_log1p": np.log1p(max(distance, 0)),
        "city_pop_log1p": np.log1p(max(city_pop, 0)),
        "transaction_day_of_week": tx_date.weekday(),
        "transaction_day": tx_date.day,
        "is_weekend": int(tx_date.weekday() >= 5),
        "is_night": int(hour <= 5 or hour >= 22),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * tx_date.month / 12),
        "month_cos": np.cos(2 * np.pi * tx_date.month / 12),
        "merchant_frequency_train": merchant_frequency,
        "job_frequency_train": job_frequency,
        "city_frequency_train": city_frequency,
        "category_frequency_train": category_frequency,
        "state_frequency_train": state_frequency,
        "card_prior_txn_count_log1p": np.log1p(prior_count),
        "card_previous_amt_log1p": np.log1p(max(previous, 0)),
        "seconds_since_previous_txn_log1p": np.log1p(seconds),
        "card_prior_mean_amt_log1p": np.log1p(max(prior_avg, 0)),
        "amt_deviation_signed_log": signed_log(amount - prior_avg) if prior else 0,
        "amt_ratio_log_clipped": np.log(ratio),
        "is_rapid_repeat_10m": int(prior and minutes_since_previous <= 10),
        "has_prior_history": int(prior),
        "category": category,
        "gender": gender,
        "state": state,
    }])


def show_image(filename, caption):
    path = ASSET_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


st.markdown("""
<div class="hero">
  <h1>🛡️ 交易詐欺風險辨識與顧客分群</h1>
  <p>以過去交易訓練、未來交易測試，並將模型結果轉為人工審查與顧客經營參考。</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "功能選單",
    ["專題總覽", "單筆交易風險評估", "模型成果", "顧客分群", "研究限制與使用方式"],
)
st.sidebar.caption("正式模型：HistGradientBoosting\n\n決策門檻：0.9345")


if page == "專題總覽":
    st.subheader("專題要解決的兩個問題")
    a, b = st.columns(2)
    with a:
        st.info("**交易層級｜詐欺風險辨識**\n\n判斷每一筆交易是否值得優先人工複核。")
    with b:
        st.success("**卡片層級｜顧客分群**\n\n將整體消費行為相近的卡片分組，提出差異化行銷建議。")

    st.subheader("資料與模型摘要")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("訓練資料", "1,296,675 筆")
    c2.metric("未來測試資料", "555,719 筆")
    c3.metric("測試集詐欺率", "0.386%")
    c4.metric("一般顧客分群", "908 張卡片")

    st.subheader("最後成果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最佳模型", "HistGradientBoosting")
    c2.metric("PR-AUC", "0.9365")
    c3.metric("Precision", "92.38%")
    c4.metric("Recall", "84.80%")
    st.caption("因詐欺交易非常少，Accuracy 容易造成誤解，因此主要使用 PR-AUC、Precision、Recall 與 F1 評估。")

elif page == "單筆交易風險評估":
    st.subheader("單筆交易風險評估")
    st.info("請輸入一筆模擬交易。結果是模型的風險分數與複核建議，不是法律判定，也不是保證正確。")

    with st.form("risk_form"):
        left, middle, right = st.columns(3)
        with left:
            amount = st.number_input("交易金額（美元）", 0.0, 50000.0, 75.0, 5.0)
            category = st.selectbox(
                "交易類別", list(CATEGORY_ZH),
                format_func=lambda x: f"{CATEGORY_ZH[x]}（{x}）",
            )
            tx_date = st.date_input("交易日期", datetime(2020, 10, 15))
            hour = st.slider("交易時間（小時）", 0, 23, 14)
        with middle:
            age = st.number_input("持卡人年齡", 18, 100, 40)
            gender = st.selectbox("性別", ["F", "M"], format_func=lambda x: "女性" if x == "F" else "男性")
            state = st.selectbox("州別（美國縮寫）", STATES, index=4)
            city_pop = st.number_input("居住城市人口", 1, 10000000, 50000, 1000)
            distance = st.number_input("持卡人與商店距離（公里）", 0.0, 10000.0, 25.0, 1.0)
        with right:
            prior_count = st.number_input("此卡過去交易次數", 0, 100000, 20)
            prior_mean = st.number_input("此卡過去平均金額", 0.0, 50000.0, 70.0, 5.0)
            previous_amount = st.number_input("上一筆交易金額", 0.0, 50000.0, 65.0, 5.0)
            minutes_since_previous = st.number_input("距上一筆交易（分鐘）", 0.0, 1000000.0, 720.0, 10.0)

        with st.expander("進階設定（一般展示可保持預設值）"):
            st.caption("這五項代表該值在訓練資料中出現的比例；新個案不易取得時，以低頻預設值模擬。")
            f1, f2, f3, f4, f5 = st.columns(5)
            merchant_frequency = f1.number_input("商家頻率", 0.0, 1.0, 0.0014, format="%.5f")
            job_frequency = f2.number_input("職業頻率", 0.0, 1.0, 0.0020, format="%.5f")
            city_frequency = f3.number_input("城市頻率", 0.0, 1.0, 0.0011, format="%.5f")
            category_frequency = f4.number_input("類別頻率", 0.0, 1.0, 0.0700, format="%.5f")
            state_frequency = f5.number_input("州別頻率", 0.0, 1.0, 0.0200, format="%.5f")

        submitted = st.form_submit_button("開始評估", type="primary", use_container_width=True)

    if submitted:
        bundle = load_model()
        features = make_features(
            amount, age, distance, city_pop, tx_date, hour, category, gender, state,
            prior_count, previous_amount, minutes_since_previous, prior_mean,
            merchant_frequency, job_frequency, city_frequency, category_frequency,
            state_frequency,
        )
        pipeline = bundle["pipeline"]
        score = float(pipeline.predict_proba(features)[0, 1])
        threshold = float(bundle.get("threshold", THRESHOLD))
        flagged = score >= threshold

        c1, c2, c3 = st.columns(3)
        c1.metric("模型風險分數", f"{score:.2%}")
        c2.metric("正式決策門檻", f"{threshold:.2%}")
        c3.metric("系統建議", "優先人工複核" if flagged else "一般流程觀察")
        if flagged:
            st.markdown(
                '<div class="result-high"><b>高於決策門檻：</b>'
                '建議交由人員檢查，但不能只靠模型直接拒絕交易。</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-low"><b>低於決策門檻：</b>'
                '目前未列入優先複核，但仍不代表完全沒有詐欺可能。</div>',
                unsafe_allow_html=True,
            )

        observations = []
        if hour <= 5 or hour >= 22:
            observations.append("本筆屬於夜間交易")
        if prior_count and minutes_since_previous <= 10:
            observations.append("與上一筆交易相隔不超過 10 分鐘")
        if prior_count and amount >= max(prior_mean * 3, prior_mean + 100):
            observations.append("金額明顯高於此卡過去平均")
        if not observations:
            observations.append("表單中的三項直觀警訊並不明顯；模型仍會綜合全部 28 個特徵評估")
        st.write("**輸入資料的直觀觀察：** " + "；".join(observations) + "。")
        st.caption("以上是輸入條件摘要，不等同於特徵因果解釋。")

elif page == "模型成果":
    st.subheader("為什麼最後選 HistGradientBoosting？")
    st.write(
        "它在未來測試資料的 PR-AUC、Precision、Recall 與 F1 都明顯優於 "
        "Logistic Regression，代表它更能抓到金額、類別、時間與過去交易行為之間的非線性組合。"
    )
    comparison = pd.DataFrame([
        ["Logistic Regression", 0.4738, 0.6242, 0.4359, 0.5133],
        ["HistGradientBoosting", 0.9365, 0.9238, 0.8480, 0.8843],
    ], columns=["模型", "PR-AUC", "Precision", "Recall", "F1"])
    st.dataframe(comparison.style.format({c: "{:.4f}" for c in comparison.columns[1:]}),
                 hide_index=True, use_container_width=True)
    st.caption("Logistic Regression 保留為可解釋的基準模型；正式應用模型選擇 HistGradientBoosting。")

    c1, c2 = st.columns(2)
    with c1:
        show_image("01_PR曲線.png", "PR 曲線：越靠近右上角越好")
    with c2:
        show_image("03_混淆矩陣.png", "測試集混淆矩陣")
    st.write("測試集共抓到 **1,819 筆**詐欺交易，漏掉 **326 筆**；誤報 **150 筆**正常交易。")
    show_image("04_特徵重要性.png", "排列重要性：金額與交易類別最重要")
    st.warning("特徵重要性表示模型依賴程度，不代表某特徵一定造成詐欺。")

elif page == "顧客分群":
    st.subheader("顧客／卡片行為分群")
    st.write(
        "第一階段先辨識出 91 張「短期異常卡片」；因其詐欺率 100% 很可能反映模擬資料設計，"
        "因此不把它當成正常行銷客群。第二階段再將 908 張一般卡片分成四群。"
    )
    st.dataframe(CLUSTER_INFO, hide_index=True, use_container_width=True)
    show_image("08_一般顧客分群.png", "908 張一般卡片的四群分布")
    st.caption("分群名稱是研究團隊依各群特徵所做的商業命名，不是 K-Means 自動產生的名稱。")

elif page == "研究限制與使用方式":
    st.subheader("怎麼確認網站預測正不正確？")
    st.write(
        "網站當下只能產生預測。等交易經過銀行調查、客訴或退款流程，得到真實的 is_fraud 標籤後，"
        "才能逐筆比對預測是否正確。正式模型的整體可信度，是用完全未參與訓練的未來測試集評估。"
    )
    st.subheader("研究限制")
    st.markdown("""
- 資料是合成資料，不能直接代表真實銀行環境。
- 測試集詐欺率僅 0.386%，類別高度不平衡，Accuracy 不是主要評估標準。
- 模型分數不是法律判定，也不是校準後的真實詐欺機率。
- 新輸入若缺少商家、城市與職業的歷史頻率，只能用近似值模擬，結果會有額外不確定性。
- 部署後若遇到資料分布改變，需要定期重新驗證與訓練。
""")
    st.subheader("正確使用方式")
    st.success("把模型當成「案件排序助手」：先找出值得人工查看的交易，再由人員綜合其他證據決定。")

st.divider()
st.caption(
    "資料來源：Kaggle Fraud Detection（合成信用卡交易資料）｜"
    "本網站供課程展示與研究使用，不應直接用於真實金融決策。"
)
