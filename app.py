
import streamlit as st
import pandas as pd
import joblib, json

st.set_page_config(page_title="Ứng dụng dự đoán giá nhà", layout="centered")
st.title("Ứng dụng dự đoán giá nhà Boston")

# ===== 1) Tải model & metadata =====
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    with open("train_columns.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta["columns"], meta.get("best_model_name", "Model đã huấn luyện")

model, train_cols, model_name = load_artifacts()

# ===== 2) Hàm xử lý đặc trưng (giống lúc train) =====
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if all(c in df2.columns for c in ["RM", "LSTAT"]):
        df2["RM_LSTAT"] = df2["RM"] * df2["LSTAT"]
        df2["RM2"] = df2["RM"] ** 2
        df2["LSTAT2"] = df2["LSTAT"] ** 2
    if all(c in df2.columns for c in ["TAX", "AGE"]):
        df2["TAX_AGE"] = df2["TAX"] * df2["AGE"]
    return df2

def ensure_train_columns(df_fe: pd.DataFrame, train_columns) -> pd.DataFrame:
    df = df_fe.copy()
    for col in train_columns:
        if col not in df.columns:
            df[col] = 0.0
    return df[train_columns]

def fmt_vnd(x: float) -> str:
    ty = x / 1_000_000_000
    trieu = x / 1_000_000
    return f"{ty:,.2f} tỷ VND (~ {trieu:,.0f} triệu)" if ty >= 1 else f"{trieu:,.0f} triệu VND"

# ===== 3) Sidebar: Tỷ giá =====
with st.sidebar:
    st.header("Tùy chọn")
    fx = st.number_input("Tỷ giá (VND / 1 USD)", min_value=15000, max_value=100000, value=25000, step=100)
    st.caption(f"Mô hình: {model_name}")

# ===== 4) Tabs: Nhập tay | CSV =====
tab1, tab2 = st.tabs(["Nhập dữ liệu thủ công", "Đọc từ file CSV"])

# ---------- TAB 1: NHẬP TAY ----------
with tab1:
    st.subheader("Nhập thông tin bất động sản")
    c1, c2, c3 = st.columns(3)
    CRIM = c1.number_input("Tỉ lệ tội phạm (CRIM)", 0.0, 100.0, 0.10, step=0.01)
    ZN   = c2.number_input("Khu đất quy hoạch (ZN)", 0.0, 100.0, 0.00, step=0.1)
    INDUS= c3.number_input("Đất công nghiệp (INDUS)", 0.0, 30.0, 6.00, step=0.1)

    c1, c2, c3 = st.columns(3)
    CHAS = c1.selectbox("Giáp sông Charles (CHAS)", [0, 1], index=0)
    NOX  = c2.number_input("Ô nhiễm NOx (NOX)", 0.0, 1.0, 0.50, step=0.01)
    RM   = c3.number_input("Số phòng trung bình (RM)", 0.0, 10.0, 6.00, step=0.1)

    c1, c2, c3 = st.columns(3)
    AGE  = c1.number_input("Tuổi nhà (AGE)", 0.0, 100.0, 60.0, step=1.0)
    DIS  = c2.number_input("Khoảng cách trung tâm (DIS)", 0.0, 20.0, 4.0, step=0.1)
    RAD  = c3.number_input("Tiếp cận cao tốc (RAD)", 1, 24, 4, step=1)

    c1, c2, c3 = st.columns(3)
    TAX  = c1.number_input("Thuế BĐS (TAX)", 0.0, 1000.0, 300.0, step=1.0)
    PTRATIO = c2.number_input("Tỉ lệ HS/GV (PTRATIO)", 5.0, 30.0, 18.0, step=0.1)
    B    = c3.number_input("Chỉ số B", 0.0, 400.0, 350.0, step=1.0)

    LSTAT = st.number_input("Tỉ lệ dân nghèo (LSTAT)", 0.0, 40.0, 12.0, step=0.1)

    if st.button("Dự đoán giá nhà"):
        x = pd.DataFrame([{
            "CRIM": CRIM, "ZN": ZN, "INDUS": INDUS, "CHAS": CHAS, "NOX": NOX,
            "RM": RM, "AGE": AGE, "DIS": DIS, "RAD": RAD, "TAX": TAX,
            "PTRATIO": PTRATIO, "B": B, "LSTAT": LSTAT
        }])
        x_ready = ensure_train_columns(make_features(x), train_cols)

        # Kết quả từ mô hình là NGHÌN USD
        y_kusd = float(model.predict(x_ready)[0])
        # Chuyển thành USD số đầy đủ để hiển thị
        y_usd  = y_kusd * 1000.0
        y_vnd  = y_usd * fx

        # HIỂN THỊ: VND + USD (không còn chữ "nghìn")
        st.success(f"Giá dự kiến: {fmt_vnd(y_vnd)} (≈ {y_usd:,.0f} USD)")

# ---------- TAB 2: CSV ----------
with tab2:
    st.subheader("Tải file CSV và dự đoán hàng loạt")
    cols_req = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]

    st.download_button(
        "Tải mẫu CSV (đúng 13 cột)",
        pd.DataFrame(columns=cols_req).to_csv(index=False).encode("utf-8-sig"),
        file_name="mau_du_lieu_gia_nha.csv", mime="text/csv"
    )
    st.caption("Không cần cột MEDV. Hãy điền số liệu đúng 13 cột gốc ở trên rồi tải lên.")

    up = st.file_uploader("Chọn file CSV", type=["csv"])

    if up is not None:
        try:
            df_raw = pd.read_csv(up)
        except Exception:
            df_raw = pd.read_csv(up, encoding="latin-1")

        st.write("Xem trước 5 dòng:")
        st.dataframe(df_raw.head())

        missing = [c for c in cols_req if c not in df_raw.columns]
        if missing:
            st.error(f"Thiếu cột: {missing}. Vui lòng dùng mẫu CSV phía trên.")
        else:
            df_ready = ensure_train_columns(make_features(df_raw[cols_req]), train_cols)

            preds_kusd = model.predict(df_ready)      # nghìn USD
            preds_usd  = preds_kusd * 1000.0          # USD số đầy đủ
            preds_vnd  = preds_usd * fx               # VND

            out = df_raw.copy()
            # MẶC ĐỊNH: giữ kUSD + VND (gọn). Nếu muốn thêm USD, bỏ comment dòng dưới.
            out["MEDV_pred_kUSD"] = preds_kusd.round(2)
            # out["Price_USD"] = preds_usd.round(0)   # <- bật nếu cần thêm cột USD
            out["Price_VND"] = preds_vnd.round(0)

            st.success(f"Đã dự đoán {len(out)} bản ghi.")
            st.dataframe(out.head())

            st.download_button(
                "Tải kết quả (CSV)",
                out.to_csv(index=False).encode("utf-8-sig"),
                file_name="du_doan_gia_nha.csv",
                mime="text/csv"
            )
