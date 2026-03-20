import streamlit as st
import numpy as np

st.set_page_config(page_title="Ти — нейромережа", page_icon="🧠", layout="centered")

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


def net_predict(features, weights, bias):
    s = bias + sum(f * w for f, w in zip(features, weights))
    return sigmoid(s)


def init_state():
    st.session_state.weights = [0.0, 0.0, 0.0]
    st.session_state.bias = 0.0
    st.session_state.round = 0
    st.session_state.phase = "guess"
    st.session_state.user_score = 0
    st.session_state.net_score = 0
    st.session_state.history = []
    st.session_state.round_result = None


def do_guess(user_guess: str):
    """Called once via on_click. Computes everything and updates state."""
    rnd = st.session_state.round
    d = DATA[rnd]
    w = list(st.session_state.weights)
    b = st.session_state.bias

    prob = net_predict(d["f"], w, b)
    net_guess = "cat" if prob > 0.5 else "dog"
    target = 1.0 if d["label"] == "cat" else 0.0
    confidence = prob if d["label"] == "cat" else 1.0 - prob

    user_correct = user_guess == d["label"]
    net_correct = net_guess == d["label"]

    if user_correct:
        st.session_state.user_score += 1
    if net_correct:
        st.session_state.net_score += 1

    err = target - prob
    grad = prob * (1.0 - prob)
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

    st.session_state.round_result = {
        "user_correct": user_correct,
        "net_correct": net_correct,
        "confidence": confidence,
        "d": d,
    }
    st.session_state.phase = "result"


def next_round():
    st.session_state.round += 1
    if st.session_state.round >= len(DATA):
        st.session_state.phase = "done"
    else:
        st.session_state.phase = "guess"
    st.session_state.round_result = None


if "phase" not in st.session_state:
    init_state()


def show_weights():
    cols = st.columns(3)
    for i, col in enumerate(cols):
        col.metric(FEATURES[i], f"{st.session_state.weights[i]:.2f}")


def show_features(d: dict):
    cols = st.columns(3)
    for i, col in enumerate(cols):
        if d["f"][i]:
            col.markdown(
                f'<div style="background:#1b4332;color:#95d5b2;padding:10px 14px;'
                f'border-radius:8px;text-align:center;font-size:15px;">'
                f"✅ {FEATURES[i]}</div>",
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f'<div style="background:#2b2b2b;color:#666;padding:10px 14px;'
                f'border-radius:8px;text-align:center;font-size:15px;">'
                f"❌ {FEATURES[i]}</div>",
                unsafe_allow_html=True,
            )


# ── GUESS ──
if st.session_state.phase == "guess":
    rnd = st.session_state.round
    d = DATA[rnd]

    st.markdown("## 🧠 Ти — нейромережа")
    st.caption(f"Раунд {rnd + 1} / {len(DATA)}")
    st.progress(rnd / len(DATA))

    if rnd == 0:
        st.markdown(
            "**Як грати:** бачиш набір ознак тварини, вгадуєш — кіт чи собака. "
            "Паралельно вгадує проста нейромережа. Після кожного раунду вона "
            "рахує свою помилку і підлаштовує ваги — точно як при справжньому навчанні."
        )
        with st.expander("Що таке ваги?"):
            st.markdown(
                "Кожна ознака (вуса, гострі вуха, муркоче) має свою **вагу** — число, "
                "яке показує, наскільки ця ознака впливає на рішення мережі.\n\n"
                "**Додатна вага** — ознака штовхає рішення в бік «кіт».\n\n"
                "**Від'ємна вага** — ознака штовхає рішення в бік «собака».\n\n"
                "**Нуль** — ознака поки не впливає на рішення.\n\n"
                "Одна й та сама ознака може бути і в кота, і в собаки. "
                "Мережа сама вчиться, які комбінації ознак найкраще розрізняють тварин. "
                "Вона множить кожну ознаку на її вагу, сумує, і отримує "
                "впевненість: ближче до 100% — кіт, ближче до 0% — собака. "
                "Якщо помилилась — ваги зсуваються в потрібний бік."
            )

    st.markdown("### Ознаки тварини:")
    show_features(d)

    st.markdown("---")
    st.markdown("**Хто це?**")
    col1, col2 = st.columns(2)
    with col1:
        st.button("🐱 Кіт", use_container_width=True, on_click=do_guess, args=("cat",))
    with col2:
        st.button("🐶 Собака", use_container_width=True, on_click=do_guess, args=("dog",))

    st.markdown("---")
    st.markdown("**Поточні ваги мережі:**")
    show_weights()


# ── RESULT ──
elif st.session_state.phase == "result":
    rnd = st.session_state.round
    r = st.session_state.round_result
    d = r["d"]

    st.markdown(f"## {d['emoji']} Це {d['ua']}!")
    st.caption(f"Раунд {rnd + 1} / {len(DATA)}")

    col1, col2 = st.columns(2)
    with col1:
        if r["user_correct"]:
            st.success(f"**Ти:** вгадав ✓\n\n{st.session_state.user_score} / {rnd + 1}")
        else:
            st.error(f"**Ти:** помилка ✗\n\n{st.session_state.user_score} / {rnd + 1}")
    with col2:
        conf_pct = f"{r['confidence'] * 100:.0f}%"
        if r["net_correct"]:
            st.success(
                f"**Мережа:** {conf_pct} впевнена ✓\n\n{st.session_state.net_score} / {rnd + 1}"
            )
        else:
            st.error(
                f"**Мережа:** {conf_pct} впевнена ✗\n\n{st.session_state.net_score} / {rnd + 1}"
            )

    st.markdown("**Оновлені ваги:**")
    show_weights()

    st.markdown("---")
    label = "Далі →" if rnd + 1 < len(DATA) else "Результати →"
    st.button(label, use_container_width=True, on_click=next_round)


# ── DONE ──
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

    w = st.session_state.weights
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
    st.button("🔄 Зіграти ще раз", use_container_width=True, on_click=init_state)
