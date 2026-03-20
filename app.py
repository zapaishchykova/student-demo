import streamlit as st
import numpy as np

st.set_page_config(page_title="Ти — нейромережа", page_icon="🧠", layout="centered")

# --- data ---
DATA = [
    {"f": [1, 1, 0], "label": "cat", "ua": "Кіт", "emoji": "🐱"},
    {"f": [0, 0, 1], "label": "dog", "ua": "Собака", "emoji": "🐶"},
    {"f": [1, 0, 0], "label": "cat", "ua": "Кіт", "emoji": "🐱"},
    {"f": [0, 1, 1], "label": "dog", "ua": "Собака", "emoji": "🐶"},
    {"f": [1, 1, 1], "label": "cat", "ua": "Кіт", "emoji": "🐱"},
    {"f": [0, 0, 0], "label": "dog", "ua": "Собака", "emoji": "🐶"},
    {"f": [1, 0, 1], "label": "cat", "ua": "Кіт", "emoji": "🐱"},
    {"f": [0, 1, 0], "label": "dog", "ua": "Собака", "emoji": "🐶"},
]
FEATURES = ["Вуса", "Гострі вуха", "Муркоче"]
LR = 0.8


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def net_predict(features: list[int], weights: list[float], bias: float) -> float:
    s = bias + sum(f * w for f, w in zip(features, weights))
    return sigmoid(s)


def init_state():
    st.session_state.weights = [0.0, 0.0, 0.0]
    st.session_state.bias = 0.0
    st.session_state.round = 0
    st.session_state.phase = "guess"
    st.session_state.user_score = 0
    st.session_state.net_score = 0
    st.session_state.last_guess = None
    st.session_state.history = []


if "phase" not in st.session_state:
    init_state()

w = st.session_state.weights
b = st.session_state.bias
rnd = st.session_state.round


# --- helpers ---
def show_weights():
    cols = st.columns(3)
    for i, col in enumerate(cols):
        col.metric(FEATURES[i], f"{w[i]:.2f}")


def show_features(d: dict):
    tags = []
    for i, name in enumerate(FEATURES):
        if d["f"][i]:
            tags.append(f"✅ {name}")
        else:
            tags.append(f"~~{name}~~")
    st.markdown("&emsp;".join(tags))


# --- screens ---
if st.session_state.phase == "guess" and rnd < len(DATA):
    d = DATA[rnd]

    st.markdown("## 🧠 Ти — нейромережа")
    st.caption(f"Раунд {rnd + 1} / {len(DATA)}")
    st.progress((rnd) / len(DATA))

    st.markdown("### Ознаки тварини:")
    show_features(d)

    st.markdown("---")
    st.markdown("**Хто це?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🐱 Кіт", use_container_width=True):
            st.session_state.last_guess = "cat"
            st.session_state.phase = "result"
            st.rerun()
    with col2:
        if st.button("🐶 Собака", use_container_width=True):
            st.session_state.last_guess = "dog"
            st.session_state.phase = "result"
            st.rerun()

    st.markdown("---")
    st.markdown("**Поточні ваги мережі:**")
    show_weights()


elif st.session_state.phase == "result":
    d = DATA[rnd]
    guess = st.session_state.last_guess

    # network prediction before update
    prob = net_predict(d["f"], w, b)
    net_guess = "cat" if prob > 0.5 else "dog"
    target = 1.0 if d["label"] == "cat" else 0.0
    confidence = prob if d["label"] == "cat" else 1 - prob

    user_correct = guess == d["label"]
    net_correct = net_guess == d["label"]

    if user_correct:
        st.session_state.user_score += 1
    if net_correct:
        st.session_state.net_score += 1

    # update weights
    err = target - prob
    grad = prob * (1 - prob)
    for i in range(3):
        st.session_state.weights[i] += LR * err * grad * d["f"][i]
    st.session_state.bias += LR * err * grad

    st.session_state.history.append(
        {
            "round": rnd + 1,
            "animal": d["ua"],
            "user": "✓" if user_correct else "✗",
            "net": "✓" if net_correct else "✗",
            "conf": f"{confidence * 100:.0f}%",
        }
    )

    # display
    st.markdown(f"## {d['emoji']} Це {d['ua']}!")
    st.caption(f"Раунд {rnd + 1} / {len(DATA)}")

    col1, col2 = st.columns(2)
    with col1:
        if user_correct:
            st.success(f"**Ти:** вгадав ✓\n\n{st.session_state.user_score} / {rnd + 1}")
        else:
            st.error(f"**Ти:** помилка ✗\n\n{st.session_state.user_score} / {rnd + 1}")
    with col2:
        if net_correct:
            st.success(
                f"**Мережа:** {confidence * 100:.0f}% впевнена ✓\n\n{st.session_state.net_score} / {rnd + 1}"
            )
        else:
            st.error(
                f"**Мережа:** {confidence * 100:.0f}% впевнена ✗\n\n{st.session_state.net_score} / {rnd + 1}"
            )

    st.markdown("**Оновлені ваги:**")
    show_weights()

    st.markdown("---")
    if rnd + 1 < len(DATA):
        if st.button("Далі →", use_container_width=True):
            st.session_state.round += 1
            st.session_state.phase = "guess"
            st.rerun()
    else:
        if st.button("Результати →", use_container_width=True):
            st.session_state.round += 1
            st.session_state.phase = "done"
            st.rerun()


elif st.session_state.phase == "done":
    st.markdown("## 🎓 Навчання завершено!")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        pct = round(st.session_state.user_score / len(DATA) * 100)
        st.metric("Ти", f"{pct}%", f"{st.session_state.user_score} / {len(DATA)}")
    with col2:
        pct = round(st.session_state.net_score / len(DATA) * 100)
        st.metric("Мережа", f"{pct}%", f"{st.session_state.net_score} / {len(DATA)}")

    st.markdown("---")
    st.markdown("**Фінальні ваги:**")
    show_weights()

    st.markdown("")
    if w[0] > 0.3 and w[1] > 0.3:
        st.info("Мережа навчилась: вуса + гострі вуха = кіт!")
    else:
        st.info("Мережа бачить патерни, але ще не ідеально.")

    st.markdown("---")
    st.markdown("**Історія раундів:**")
    for h in st.session_state.history:
        st.markdown(
            f"Раунд {h['round']}: {h['animal']} — "
            f"ти {h['user']}, мережа {h['net']} ({h['conf']})"
        )

    st.markdown("")
    if st.button("🔄 Зіграти ще раз", use_container_width=True):
        init_state()
        st.rerun()