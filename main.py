import streamlit as st
from pixel_sort import main, COLOR_SPACE_DICT

COLOR_SPACE_TUPLE = tuple(COLOR_SPACE_DICT.keys())

YCbCrch = {0: "Y", 1: "Cr", 2: "Cb"}


def make_format(index):
    if select_color_space == "YCrCb":
        return YCbCrch[index]
    return select_color_space[index]


st.set_page_config("ピクセルソート")

with st.sidebar:
    image = st.file_uploader("画像アップロード", accept_multiple_files=False)
    with st.expander("チャンネル選択", True):
        select_color_space = st.selectbox(
            "色空間", COLOR_SPACE_TUPLE, help="どの色空間をソートに使うか"
        )
        select_channel = st.radio(
            "チャンネル",
            range(3),
            help="どのチャンネルを基準にソートさせるか",
            format_func=make_format,
            horizontal=True,
        )
    angle = st.slider("ソート角度", min_value=0, max_value=360)
    with st.expander("閾値の設定", True):
        select_range_lower, select_range_upper = st.slider(
            "閾値", min_value=0, max_value=255, value=(50, 200)
        )
        select_sort_target = st.radio(
            "ソートさせる部分",
            range(3),
            format_func=lambda x: ["閾値の範囲内", "閾値の範囲外", "両方"][x],
            horizontal=True,
        )

    ispolar = st.checkbox("極座標")
    if ispolar:
        polar_degrees = st.slider(
            "極座標の始端と終端の角度",
            min_value=0,
            max_value=360,
            help="上にある「ソート角度」が0、180、360に近いほど効果が分かりやすく、90、270に近いほど分かりにくい",
        )
    else:
        polar_degrees = None

    clicked = st.button("実行")

if clicked and image:
    with st.spinner("処理中…"):
        image = main(
            image,
            None,
            select_color_space,
            select_channel,
            select_sort_target,
            select_range_lower,
            select_range_upper,
            angle,
            ispolar,
            polar_degrees,
        )

if image:
    st.image(image)
