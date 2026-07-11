
"""
흐름 B: 기질 강성이 세포 운명을 정한다 (2011년 YAP/TAZ 발견 이전, 메커니즘 미지)
------------------------------------------------------------------------------
역사적 배경:
- Pelham & Wang (1997, PNAS): 세포가 기질의 유연성에 따라 다르게 움직인다는 최초 관찰
- Engler et al. (2006, Cell) "Matrix Elasticity Directs Stem Cell Lineage Specification":
  부드러운 기질(뇌 유사, ~0.1-1 kPa) -> 신경세포, 중간 강성(근육 유사, ~8-17 kPa) -> 근육세포,
  단단한 기질(뼈 유사, ~25-40 kPa) -> 뼈세포로 중간엽줄기세포(MSC)가 분화한다는 걸 발견.
  저자들은 비근육 미오신 기반 세포골격 장력이 관여할 것으로 추정했지만,
  "장력을 감지해 유전자 발현으로 바꾸는 분자가 무엇인지"는 이 시점까지 알려지지 않았음
  (이 고리가 2011년 Dupont et al.의 YAP/TAZ 발견으로 채워짐).

이 앱은 이 "미지의 연결고리" 상태 그대로, 순수 현상론적(phenomenological) 모델로
기질 강성 -> 세포골격 장력 -> 분화 확률의 관계를 재현합니다. 여기엔 흐름 A와 달리
검증된 생화학적 인산화 경로가 없고, 관찰된 상관관계만 존재한다는 점이 핵심입니다.

모델:
  tension(E)      = E / (E + Km)                         (장력, 강성에 따라 포화적으로 증가)
  score_i(E)      = exp(-(log10(E) - mu_i)^2 / (2*sigma^2))   (세 가지 계통별 적합도)
  P_i(E)          = score_i(E) / sum(score)                (정규화된 분화 확률)

실행 방법: pip install -r requirements_flowb.txt
           streamlit run flow_b_stiffness_app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="흐름 B: 기질 강성 시뮬레이터", layout="wide")

st.title("🧫 흐름 B: 기질 강성이 정하는 줄기세포 운명 (메커니즘 미지 시대)")
st.markdown("""
2011년 YAP/TAZ가 발견되기 전, **기질이 단단할수록 세포골격 장력이 커지고 특정 계통으로
분화한다**는 현상은 관찰되었지만, 장력을 유전자 발현으로 바꾸는 분자가 무엇인지는
알려지지 않았습니다. 기질 강성을 조절하며 그 '현상'만 먼저 확인해보세요.
""")

# ---------------- 사이드바 ----------------
st.sidebar.header("① 기질 강성")
E = st.sidebar.slider(
    "기질 강성 E (kPa, 로그 스케일)", 0.1, 40.0, 8.0, 0.1,
    help="0.1~1 kPa: 뇌 유사(연함) / 8~17 kPa: 근육 유사 / 25~40 kPa: 뼈 유사(단단함)"
)

st.sidebar.header("② 모델 매개변수")
Km = st.sidebar.slider("장력 민감도 Km (kPa)", 2.0, 30.0, 10.0, 1.0,
                        help="이 값이 작을수록 낮은 강성에서도 장력이 빨리 포화됨")
sigma = st.sidebar.slider("계통 적합도 폭 sigma (log10 단위)", 0.15, 0.7, 0.35, 0.01,
                           help="값이 클수록 세 계통 사이 경계가 흐려짐(분화 결정이 덜 뚜렷함)")

# ---------------- 모델 계산 ----------------
def tension(E, Km):
    return E / (E + Km)

def lineage_probs(E, sigma):
    logE = np.log10(E)
    mus = {"neuro": np.log10(0.5), "muscle": np.log10(11), "bone": np.log10(30)}
    scores = {k: np.exp(-((logE - mu) ** 2) / (2 * sigma ** 2)) for k, mu in mus.items()}
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()}

t_val = tension(E, Km)
probs = lineage_probs(E, sigma)
winner = max(probs, key=probs.get)

winner_kr = {"neuro": "신경세포 (연한 기질)", "muscle": "근육세포 (중간 강성)", "bone": "뼈세포 (단단한 기질)"}
winner_color = {"neuro": "#6fb3a8", "muscle": "#e7b84e", "bone": "#e2725b"}

# ---------------- 흐름도 (HTML/CSS) ----------------
tension_bar = t_val * 100

diagram_html = f"""
<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css");
  @import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap");
  body{{ margin:0; background:#0d2426; font-family:"Pretendard",sans-serif; color:#eef2ea; padding:14px 6px; }}
  .box{{ border-radius:12px; padding:12px 14px; border:1px solid #2a5a5d; background:#163d40; margin-bottom:10px; }}
  .tag{{ font-family:"JetBrains Mono",monospace; font-size:10px; color:#a9c2bd; letter-spacing:.06em; }}
  .title{{ font-weight:700; font-size:15px; margin:3px 0 2px; }}
  .val{{ font-size:12.5px; color:#a9c2bd; }}
  .arrow{{ font-size:18px; color:#e7b84e; text-align:center; margin: 2px 0; }}
  .bar-track{{ height:8px; border-radius:4px; background:#0d2426; margin-top:6px; overflow:hidden; }}
  .bar-fill{{ height:100%; border-radius:4px; }}
  .unknown-box{{
    border-radius:12px; padding:16px; text-align:center; margin-bottom:10px;
    border:2px dashed #6b7280; background:repeating-linear-gradient(45deg, #17282a, #17282a 8px, #1c3235 8px, #1c3235 16px);
  }}
  .unknown-q{{ font-size:26px; font-weight:800; color:#9aa5a3; }}
  .unknown-label{{ font-size:12.5px; color:#9aa5a3; margin-top:4px; }}
  .final-box{{ border-radius:12px; padding:14px; border:2px solid {winner_color[winner]}; }}
  .fate-row{{ display:flex; gap:8px; margin-top:8px; }}
  .fate{{ flex:1; text-align:center; border-radius:10px; padding:8px 4px; border:1px solid #2a5a5d; opacity:0.5; }}
  .fate.win{{ opacity:1; border-color: currentColor; }}
  .fate .name{{ font-size:12px; font-weight:700; }}
  .fate .pct{{ font-family:"JetBrains Mono",monospace; font-size:13px; margin-top:2px; }}
</style></head>
<body>

  <div class="box">
    <div class="tag">1 · 기질 강성 (측정된 물리량)</div>
    <div class="title">E = {E:.1f} kPa</div>
    <div class="val">{'단단함 (뼈 유사)' if E>20 else '중간 (근육 유사)' if E>4 else '부드러움 (뇌 유사)'}</div>
    <div class="bar-track"><div class="bar-fill" style="width:{min(E/40*100,100):.0f}%; background:#e7b84e"></div></div>
  </div>

  <div class="arrow">↓ (Pelham &amp; Wang 1997, RhoA-ROCK-미오신)</div>

  <div class="box">
    <div class="tag">2 · 세포골격 장력 (관찰된 상관관계)</div>
    <div class="title">actomyosin tension = {t_val:.2f}</div>
    <div class="bar-track"><div class="bar-fill" style="width:{tension_bar:.0f}%; background:#e2725b"></div></div>
  </div>

  <div class="arrow">↓</div>

  <div class="unknown-box">
    <div class="unknown-q">?</div>
    <div class="unknown-label">장력을 유전자 발현으로 바꾸는 분자 — 2011년 이전엔 미지<br>(2011년 Dupont et al.이 YAP/TAZ로 밝힘)</div>
  </div>

  <div class="arrow">↓ (Engler et al. 2006, 관찰된 현상)</div>

  <div class="box final-box" style="color:{winner_color[winner]}">
    <div class="tag">3 · 관찰된 분화 결과 (현상론적)</div>
    <div class="title">{winner_kr[winner]}</div>
    <div class="fate-row">
      <div class="fate {'win' if winner=='neuro' else ''}" style="color:#6fb3a8">
        <div class="name">신경</div><div class="pct">{probs['neuro']*100:.0f}%</div>
      </div>
      <div class="fate {'win' if winner=='muscle' else ''}" style="color:#e7b84e">
        <div class="name">근육</div><div class="pct">{probs['muscle']*100:.0f}%</div>
      </div>
      <div class="fate {'win' if winner=='bone' else ''}" style="color:#e2725b">
        <div class="name">뼈</div><div class="pct">{probs['bone']*100:.0f}%</div>
      </div>
    </div>
  </div>

</body></html>
"""

components.html(diagram_html, height=560, scrolling=False)

# ---------------- 그래프 ----------------
st.subheader("기질 강성에 따른 분화 확률 곡선")
E_range = np.logspace(np.log10(0.1), np.log10(40), 300)
probs_range = {"neuro": [], "muscle": [], "bone": []}
for e in E_range:
    p = lineage_probs(e, sigma)
    for k in probs_range:
        probs_range[k].append(p[k])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(E_range, probs_range["neuro"], color="#6fb3a8", label="신경세포 (연함)")
ax.plot(E_range, probs_range["muscle"], color="#e7b84e", label="근육세포 (중간)")
ax.plot(E_range, probs_range["bone"], color="#e2725b", label="뼈세포 (단단함)")
ax.axvline(E, color="white", linestyle=":", linewidth=1.2)
ax.set_xscale("log")
ax.set_xlabel("기질 강성 E (kPa, 로그축)")
ax.set_ylabel("분화 확률")
ax.legend()
st.pyplot(fig)

# ---------------- 설명 ----------------
st.markdown(f"""
---
**지금 그래프가 보여주는 것**

- 세 곡선은 Engler et al.(2006)이 실제로 관찰한 패턴을 재현한 것입니다: 기질이 부드러울수록
  신경세포, 중간 강성에서는 근육세포, 단단할수록 뼈세포로 분화하는 경향이 나타납니다.
- 흰 점선은 지금 슬라이더로 설정한 E={E:.1f} kPa 지점이고, 현재 가장 유력한 계통은
  **{winner_kr[winner]}** 입니다({probs[winner]*100:.0f}%).
- 위쪽 흐름도의 **점선 물음표 상자**는 일부러 비워둔 것입니다. 이 시기(2011년 이전)에는
  '① 강성이 세포골격 장력을 높인다'는 것과 '② 장력이 특정 분화 결과와 상관관계가 있다'는
  것까지는 실험적으로 알려졌지만, **①과 ②를 잇는 구체적인 분자(전사인자)는 미지수**였습니다.
- 흐름 A(Hippo 경로)와 비교하면 차이가 뚜렷합니다: 흐름 A는 Hpo-Sav-Wts-Mats라는
  **검증된 인산화 경로**가 있었던 반면, 흐름 B는 이 시점까지 **현상은 있지만 경로는 없는**
  상태였습니다. sigma 슬라이더를 키워보면 세 계통 사이 경계가 흐려지는데, 이는 당시
  '왜 어떤 세포는 예외적으로 다르게 반응하는지' 설명하기 어려웠던 상황과도 비슷합니다.
""")
