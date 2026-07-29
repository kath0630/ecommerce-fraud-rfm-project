from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

import joblib
import pandas as pd
import streamlit as st

RESULT_DIR = APP_DIR

st.set_page_config(
    page_title="電商風險與顧客分群展示",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource
def load_models():
    fraud_bundle = joblib.load(RESULT_DIR / "best_fraud_model.joblib")
    kmeans_bundle = joblib.load(RESULT_DIR / "rfm_kmeans_model.joblib")
    return fraud_bundle, kmeans_bundle


@st.cache_data
def load_results():
    metrics = pd.read_csv(RESULT_DIR / "fraud_model_metrics.csv")
    segments = pd.read_csv(RESULT_DIR / "rfm_segment_counts.csv")
    clusters = pd.read_csv(RESULT_DIR / "rfm_cluster_profiles.csv")
    return metrics, segments, clusters


fraud_bundle, kmeans_bundle = load_models()
metrics, segments, clusters = load_results()

st.title("電商交易詐欺風險與顧客分群")
st.caption("機器學習成果展示原型｜預測結果僅供風險排序與人工複核")

page = st.sidebar.radio(
    "選擇功能",
    ["專題成果總覽", "單筆交易風險預測", "RFM 顧客分群"],
)

if page == "專題成果總覽":
    st.header("專題成果總覽")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("交易筆數", "150,000")
    c2.metric("有交易的顧客", "24,938")
    c3.metric("詐欺交易", "6,331")
    c4.metric("整體詐欺率", "4.22%")

    st.subheader("模型結果")
    c1, c2, c3 = st.columns(3)
    c1.metric("最佳模型", "Logistic Regression")
    c2.metric("PR-AUC", "0.0666")
    c3.metric("高風險名單詐欺率", "7.45%", "約為整體的 1.8 倍")
    st.info(
        "模型能力屬於初步風險排序：它能把較可疑的交易排到前面，"
        "但不適合直接自動拒絕交易。"
    )

    st.subheader("RFM 顧客類型")
    name_map = {
        "Loyal High Value": "忠誠高價值客",
        "Hibernating": "沉睡顧客",
        "Recent Potential": "近期潛力客",
        "Champions": "冠軍顧客",
        "At Risk High Value": "高價值流失風險客",
        "Regular": "一般顧客",
    }
    chart_data = segments.copy()
    chart_data["顧客類型"] = chart_data["rfm_segment"].map(name_map)
    chart_data["顧客占比（%）"] = chart_data["customer_share"] * 100
    st.bar_chart(chart_data.set_index("顧客類型")["顧客占比（%）"])

elif page == "單筆交易風險預測":
    st.header("單筆交易風險預測")
    st.write("請輸入交易當下可以取得的資料。")

    left, right = st.columns(2)
    with left:
        order_value = st.number_input("訂單金額", min_value=0.0, value=300.0)
        payment_method = st.selectbox(
            "付款方式",
            ["Credit Card", "Debit Card", "PayPal", "UPI", "Crypto"],
        )
        device_type = st.selectbox("裝置類型", ["Desktop", "Mobile", "Tablet"])
        discount_applied = st.selectbox(
            "是否使用折扣",
            [0, 1],
            format_func=lambda x: "是" if x == 1 else "否",
        )
        category = st.selectbox(
            "商品類別",
            ["Books", "Electronics", "Fashion", "Home & Kitchen",
             "Beauty", "Sports", "Toys"],
        )
        price = st.number_input("商品價格", min_value=0.0, value=250.0)
    with right:
        margin_percentage = st.number_input(
            "毛利率", min_value=0.0, max_value=100.0, value=25.0
        )
        popularity_score = st.number_input(
            "商品熱門度", min_value=0.0, max_value=100.0, value=50.0
        )
        age = st.number_input("顧客年齡", min_value=18, max_value=100, value=35)
        gender = st.selectbox("性別", ["Female", "Male", "Other"])
        country = st.selectbox(
            "國家",
            ["USA", "Canada", "UK", "Germany", "France", "India", "Brazil"],
        )

    if st.button("開始預測", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "order_value": order_value,
            "payment_method": payment_method,
            "device_type": device_type,
            "discount_applied": discount_applied,
            "category": category,
            "price": price,
            "margin_percentage": margin_percentage,
            "popularity_score": popularity_score,
            "age": age,
            "gender": gender,
            "country": country,
        }])
        probability = float(
            fraud_bundle["pipeline"].predict_proba(row)[0, fraud_bundle["positive_class"]]
        )
        threshold = float(fraud_bundle["threshold"])
        if probability >= threshold:
            level, color = "高風險", "error"
        elif probability >= threshold * 0.75:
            level, color = "中風險", "warning"
        else:
            level, color = "低風險", "success"

        c1, c2, c3 = st.columns(3)
        c1.metric("詐欺風險分數", f"{probability:.1%}")
        c2.metric("風險等級", level)
        c3.metric("模型門檻", f"{threshold:.2%}")
        getattr(st, color)(
            "建議：高風險交易應交由人工進一步確認。"
            "模型分數不代表此交易一定是詐欺。"
        )

else:
    st.header("RFM 顧客分群")
    st.write("輸入顧客的最近消費、消費次數與累積金額。")

    recency = st.number_input("距離最近一次消費的天數（R）", min_value=0, value=180)
    frequency = st.number_input("消費次數（F）", min_value=1, value=6)
    monetary = st.number_input("累積消費金額（M）", min_value=0.0, value=2500.0)

    if st.button("查看顧客類型", type="primary", use_container_width=True):
        import numpy as np

        features = pd.DataFrame([{
            "log_recency": np.log1p(recency),
            "log_frequency": np.log1p(frequency),
            "log_monetary": np.log1p(monetary),
        }])
        cluster = int(
            kmeans_bundle["model"].predict(
                kmeans_bundle["scaler"].transform(features)
            )[0]
        )
        cluster_name = "活躍高價值群" if cluster == 0 else "較久未購低頻群"

        if recency <= 180 and frequency >= 8 and monetary >= 3000:
            rfm_type, advice = "冠軍顧客", "提供 VIP 會員升級、專屬優惠或推薦獎勵。"
        elif frequency >= 6 and monetary >= 2500:
            rfm_type, advice = "忠誠高價值客", "維持關係並提供會員專屬活動。"
        elif recency <= 180:
            rfm_type, advice = "近期潛力客", "提供第二次購買優惠，鼓勵形成消費習慣。"
        elif monetary >= 2500:
            rfm_type, advice = "高價值流失風險客", "優先提供召回優惠與會員關懷。"
        elif recency >= 365:
            rfm_type, advice = "沉睡顧客", "使用低成本再行銷，測試顧客是否願意回購。"
        else:
            rfm_type, advice = "一般顧客", "維持一般促銷溝通並觀察後續消費。"

        c1, c2 = st.columns(2)
        c1.metric("K-Means 群組", cluster_name)
        c2.metric("RFM 顧客類型", rfm_type)
        st.success(f"建議：{advice}")

st.divider()
st.caption(
    "資料來源：Kaggle Enterprise E-Commerce Intelligence｜"
    "本介面為期末專題展示原型，非正式企業風控系統。"
)
