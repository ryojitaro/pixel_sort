import streamlit as st
from pixel_sort import pixel_sort

color_space_tuple = ("RGB", "HSV", "HLS", "YCrCb", "Lab", "Luv", "XYZ")


def make_format(index):
    if select_color_space == "YCrCb":
        match index:
            case 0:
                return "Y"
            case 1:
                return "Cr"
            case 2:
                return "Cb"
    return select_color_space[index]


st.set_page_config("ピクセルソート")

with st.sidebar:
    image = st.file_uploader("画像アップロード", accept_multiple_files=False)
    angle = st.slider("角度", min_value=0, max_value=360)
    with st.expander("チャンネル選択"):
        select_color_space = st.selectbox(
            "色空間", color_space_tuple, help="どの色空間をソートに使うか"
        )
        select_channel = st.radio(
            "チャンネル",
            range(3),
            help="どのチャンネルを基準にソートさせるか",
            format_func=make_format,
            horizontal=True,
        )
    with st.expander("閾値の設定"):
        select_range_lower, select_range_upper = st.slider(
            "閾値", min_value=0, max_value=255, value=(50, 200)
        )
        select_threshold_mode = st.radio(
            "ソートさせる部分",
            range(3),
            format_func=lambda x: ["閾値の範囲内", "閾値の範囲外", "両方"][x],
            horizontal=True,
        )

    clicked = st.button("実行")

if clicked and image:
    with st.spinner("処理中…"):
        image = pixel_sort(
            image,
            select_color_space,
            select_channel,
            select_threshold_mode,
            select_range_lower,
            select_range_upper,
            angle,
        )

if image:
    st.image(image)
