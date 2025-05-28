import streamlit as st

from pixel_sort import COLOR_SPACE_DICT, SORT_TARGETS_DICT, PixelSort, PixelsortConfig

COLOR_SPACE_TUPLE = tuple(COLOR_SPACE_DICT.keys())

YCbCrch = {0: "Y", 1: "Cr", 2: "Cb"}

st.set_page_config("ピクセルソート")

with st.sidebar:
    image = st.file_uploader("画像アップロード", accept_multiple_files=False)
    with st.expander("チャンネル選択", True):
        color_space = st.selectbox(
            "色空間", COLOR_SPACE_TUPLE, help="どの色空間をソートに使うか"
        )
        channel = st.radio(
            "チャンネル",
            range(3),
            help="どのチャンネルを基準にソートさせるか",
            format_func=lambda index: YCbCrch[index]
            if color_space == "YCrCb"
            else color_space[index],
            horizontal=True,
        )
    angle = st.slider("ソート角度", min_value=0, max_value=360)
    with st.expander("閾値の設定", True):
        range_lower, range_upper = st.slider(
            "閾値", min_value=0, max_value=255, value=(50, 200)
        )
        sort_targets = st.radio(
            "ソートさせる部分",
            SORT_TARGETS_DICT.keys(),
            format_func=lambda x: {
                "in": "閾値の範囲内",
                "out": "閾値の範囲外",
                "both": "両方",
            }[x],
            horizontal=True,
        )

    ispolar = st.checkbox("極座標")
    if ispolar:
        polar_deg = st.slider(
            "極座標の始端と終端の角度",
            min_value=0,
            max_value=360,
            help="上にある「ソート角度」が0、180、360に近いほど効果が分かりやすく、90、270に近いほど分かりにくい",
        )
    else:
        polar_deg = 0.0

    cfg = PixelsortConfig(
        color_space,
        channel,
        sort_targets,
        range_lower,
        range_upper,
        angle,
        ispolar,
        polar_deg,
    )

if image:
    sorter = PixelSort(image, cfg)
    with st.spinner("処理中…"):
        image = sorter.main()

if image:
    st.image(image)
